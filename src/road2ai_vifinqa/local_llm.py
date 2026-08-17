"""Reproducible client and process manager for the local open-weight LLM."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .paths import ARTIFACT_ROOT, PROJECT_ROOT


# -------------------------------------------------------------------------
# Cloud API Key Configuration (e.g. Groq, Gemini, OpenAI, DeepInfra)
# You can set LLM_API_KEY in your environment, or paste your API key below:
# GROQ Free Key: https://console.groq.com/
# Gemini Free Key: https://aistudio.google.com/
# -------------------------------------------------------------------------
API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("GROQ_API_KEY", os.environ.get("GEMINI_API_KEY", "")))

RUNTIME = PROJECT_ROOT / "tools" / "llama.cpp" / "runtime" / "llama-server.exe"
MODEL = Path(
    os.environ.get(
        "VIFINQA_MODEL",
        str(ARTIFACT_ROOT / "models" / "Qwen3-4B-GGUF" / "Qwen3-4B-Q4_K_M.gguf"),
    )
) if not (os.getenv("USE_OLLAMA") or API_KEY) else None

if API_KEY:
    DEFAULT_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    MODEL_SOURCE = os.environ.get("LLM_MODEL", os.environ.get("VIFINQA_MODEL_SOURCE", "qwen-2.5-32b"))
elif os.getenv("USE_OLLAMA"):
    DEFAULT_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
    MODEL_SOURCE = os.environ.get("VIFINQA_MODEL_SOURCE", "qwen2.5:latest")
else:
    DEFAULT_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8087")
    MODEL_SOURCE = os.environ.get("VIFINQA_MODEL_SOURCE", MODEL.name if MODEL else "qwen2.5:latest")


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float


def server_ready(base_url: str = DEFAULT_URL) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_server(*, base_url: str = DEFAULT_URL, timeout: float = 180.0) -> subprocess.Popen[str] | None:
    """Start one hidden local server; return None when one is already ready."""

    if server_ready(base_url):
        return None
    if os.getenv("USE_OLLAMA"):
        return None  # No local server needed when using Ollama
    if not RUNTIME.exists():
        raise FileNotFoundError(RUNTIME)
    if not MODEL or not MODEL.exists() or MODEL.stat().st_size < 2_000_000_000:
        raise FileNotFoundError(f"Model checkpoint is absent or incomplete: {MODEL}")
    logs = MODEL.parent
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / "server.stdout.log").open("a", encoding="utf-8")
    stderr = (logs / "server.stderr.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(RUNTIME),
            "-m",
            str(MODEL),
            "--host",
            "127.0.0.1",
            "--port",
            base_url.rsplit(":", 1)[-1],
            "-ngl",
            "99",
            "-c",
            "16384",
            "--parallel",
            "1",
            "--jinja",
        ],
        cwd=PROJECT_ROOT,
        stdout=stdout,
        stderr=stderr,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}; see {stderr.name}")
        if server_ready(base_url):
            return process
        time.sleep(1)
    process.terminate()
    raise TimeoutError("Local LLM server did not become ready")


def chat(
    *,
    system: str,
    user: str,
    base_url: str = DEFAULT_URL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Completion:
    # Qwen3 enables a hidden reasoning mode by default.  Disabling it here keeps
    # structured generations short and, importantly, prevents the reasoning
    # budget from consuming all ``max_tokens`` before the JSON answer is emitted.
    system = f"{system.rstrip()}\n/no_think"
    user = f"{user.rstrip()}\n/no_think"
    payload = json.dumps(
        {
            "model": MODEL_SOURCE,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    # Determine the correct endpoint based on whether Ollama is used
    endpoint = f"{base_url}/api/chat" if (os.getenv("USE_OLLAMA") and not API_KEY) else f"{base_url.rstrip('/')}/v1/chat/completions" if not base_url.endswith("/v1") else f"{base_url.rstrip('/')}/chat/completions"
    if not endpoint.startswith("http"):
        endpoint = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    started = time.time()
    result = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as err:
            if err.code in (429, 503) and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise

    if result is None:
        raise RuntimeError("No response from LLM server")

    usage = result.get("usage", {})
    if "choices" in result and result["choices"]:
        content = result["choices"][0]["message"]["content"]
    elif "message" in result and isinstance(result["message"], dict):
        content = result["message"]["content"]
    else:
        content = result.get("content", "")
    return Completion(
        content=content,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        elapsed_seconds=time.time() - started,
    )


def extract_json(text: str) -> dict[str, object]:
    """Extract the first balanced JSON object from a model response."""

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unbalanced JSON object in model response")
