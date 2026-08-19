"""
Financial Report QA - Smart Conversational AI Assistant (Full Pipeline Demo)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page Config & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinReport AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .chat-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .chat-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .chat-header p {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .answer-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 5px solid #10b981;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
    }

    .answer-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Resources (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="📂 Đang khởi tạo hệ thống RAG & Full Pipeline...")
def load_resources():
    sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))
    from road2ai_vifinqa.corpus import Corpus
    from road2ai_vifinqa.panel import FinancialPanel
    from road2ai_vifinqa.template_solver import TemplateSolver

    corpus = Corpus()
    panel = FinancialPanel()
    template = TemplateSolver(corpus, panel)
    return corpus, panel, template


@st.cache_data(show_spinner=False)
def load_sample_questions():
    questions_path = Path(__file__).resolve().parents[0] / "data" / "vifinqa" / "questions" / "questions.jsonl"
    if not questions_path.exists():
        return []
    with open(questions_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


try:
    corpus, panel, template_solver = load_resources()
    resources_ok = True
except Exception as e:
    resources_ok = False
    load_error = str(e)

sample_questions = load_sample_questions()
question_by_text = {q["question"].strip(): q["id"] for q in sample_questions}

# ---------------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/bot.png", width=64)
    st.title("🤖 Cấu hình Trợ lý AI")

    llm_mode = st.radio(
        "Động cơ LLM",
        ["Cloud API (Groq/Gemini)", "Local Ollama"],
        index=0,
    )

    if llm_mode == "Cloud API (Groq/Gemini)":
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Dán API Key (gsk_... hoặc AIzaSy...)",
            help="Lấy API Key FREE tại console.groq.com hoặc aistudio.google.com",
        )
        provider = st.selectbox(
            "Provider",
            ["Groq Cloud", "Google Gemini", "OpenAI / Khác"],
        )
        
        if provider == "Groq Cloud":
            base_url = "https://api.groq.com/openai/v1"
            model_name = st.selectbox("Model", ["qwen-2.5-32b", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"])
        elif provider == "Google Gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        else:
            base_url = st.text_input("Base URL", "https://api.openai.com/v1")
            model_name = st.text_input("Model Name", "gpt-4o-mini")

        if api_key:
            os.environ["LLM_API_KEY"] = api_key
            os.environ["LLM_BASE_URL"] = base_url
            os.environ["LLM_MODEL"] = model_name
    else:
        ollama_model = st.text_input("Model Ollama", value="qwen2.5:latest")
        os.environ["USE_OLLAMA"] = "1"
        os.environ["VIFINQA_MODEL_SOURCE"] = ollama_model

    st.markdown("---")
    st.markdown("### 📊 Thống kê Kho dữ liệu")
    st.markdown("• **Công ty niêm yết:** 100 mã")
    st.markdown("• **Báo cáo tài chính:** 1.973 tệp")
    st.markdown("• **Số bảng dữ liệu:** 146.246 bảng")
    st.markdown("• **Thời gian:** 2015 – 2025")

    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header UI
# ---------------------------------------------------------------------------
st.markdown("""
<div class="chat-header">
    <h1>📈 FinReport AI Assistant</h1>
    <p>Trợ lý AI Trích xuất & Phân tích Báo cáo Tài chính tự động cho 100 Doanh nghiệp Niêm yết Việt Nam</p>
</div>
""", unsafe_allow_html=True)

if not resources_ok:
    st.error(f"❌ Không thể khởi động hệ thống dữ liệu: {load_error}\n\nHãy đảm bảo đã chạy build index trước.")
    st.stop()

# ---------------------------------------------------------------------------
# Session State Chat History
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi là Trợ lý AI chuyên nghiệp phân tích Báo cáo Tài chính doanh nghiệp Việt Nam. Hãy đặt câu hỏi về doanh thu, lợi nhuận, tài sản hay các chỉ số tài chính của 100 công ty niêm yết (VNM, FPT, VIC, HPG, VJC,...)",
        }
    ]

# Render Quick Action Chips / Suggested Prompts
st.markdown("**💡 Thử hỏi các câu hỏi mẫu:**")
chip_cols = st.columns(4)

suggested_queries = [
    "Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là bao nhiêu triệu đồng?",
    "Số dư cho vay khách hàng ngành Thương mại của ACB cuối năm 2022 là bao nhiêu triệu đồng?",
    "Doanh thu thuần của FPT năm 2023 là bao nhiêu tỷ đồng?",
    "🎲 Lấy câu hỏi ngẫu nhiên từ Dataset",
]

prompt_to_trigger = None

for i, query in enumerate(suggested_queries):
    with chip_cols[i]:
        if st.button(query, key=f"chip_{i}", use_container_width=True):
            if "🎲" in query and sample_questions:
                q_obj = random.choice(sample_questions)
                prompt_to_trigger = q_obj["question"]
            else:
                prompt_to_trigger = query

# ---------------------------------------------------------------------------
# Render Chat History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        if isinstance(msg["content"], str):
            st.markdown(msg["content"])
        else:
            data = msg["content"]
            st.markdown(data.get("text_summary", ""))

            if "answer" in data:
                st.markdown(f"""
                <div class="answer-card">
                    <span style="color:#94a3b8;font-size:0.9rem;">KẾT QUẢ TÍNH TOÁN / TRA CỨU:</span><br>
                    <span class="answer-value">{data['answer']:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)

            if data.get("docs"):
                st.markdown("**📄 Báo cáo tài chính liên quan:**")
                badges_html = "".join([f'<span class="metric-badge">📂 {doc}</span>' for doc in data["docs"][:4]])
                st.markdown(badges_html, unsafe_allow_html=True)

            if data.get("evidence"):
                with st.expander("📋 Xem Bảng Bằng Chứng Dữ Liệu (Evidence Tables)", expanded=False):
                    for ev in data["evidence"]:
                        st.caption(f"Biến: `{ev['variable']}`")
                        st.dataframe(ev["frame"], use_container_width=True)

            if data.get("pandas_query"):
                with st.expander("🔢 Mã lệnh Pandas tính toán", expanded=False):
                    st.code(data["pandas_query"], language="python")

            if data.get("elapsed"):
                st.caption(f"⚡ Thời gian phản hồi: `{data['elapsed']:.2f}s` | Tuyến xử lý: `{data.get('route', 'Auto')}` | Phương pháp: `{data.get('method', 'Full Pipeline')}`")

# ---------------------------------------------------------------------------
# Smart Multi-Route Solver Logic (Matches solve.py 100%)
# ---------------------------------------------------------------------------
def run_smart_pipeline(question: str):
    from road2ai_vifinqa.solve import route_for_id
    from road2ai_vifinqa.pipeline import (
        solve_easy_submission,
        solve_hard_submission,
        solve_note_submission,
        solve_template_submission,
    )

    q_clean = question.strip()
    qid = question_by_text.get(q_clean, 9999)

    # 1. Known Dataset Question -> Use exact route
    if qid != 9999:
        route = route_for_id(qid)
        if route == "direct":
            return solve_easy_submission(qid, q_clean, corpus, max_attempts=3, log_path=None), route
        elif route == "hard":
            return solve_hard_submission(qid, q_clean, panel), route
        elif route == "note":
            return solve_note_submission(qid, q_clean, corpus, max_attempts=3, log_path=None), route
        elif route == "template":
            return solve_template_submission(qid, q_clean, template_solver), route

    # 2. Custom User Question -> Smart Dynamic Fallback Cascade
    # Priority: Template -> Hard Formula -> Note -> Easy LLM
    try:
        sol = solve_template_submission(9999, q_clean, template_solver)
        return sol, "template"
    except Exception:
        pass

    try:
        sol = solve_hard_submission(9999, q_clean, panel)
        return sol, "hard"
    except Exception:
        pass

    try:
        sol = solve_note_submission(9999, q_clean, corpus, max_attempts=3, log_path=None)
        return sol, "note"
    except Exception:
        pass

    # Final fallback to Direct / Easy LLM solver
    sol = solve_easy_submission(9999, q_clean, corpus, max_attempts=3, log_path=None)
    return sol, "direct"

# ---------------------------------------------------------------------------
# Chat Input & Execution
# ---------------------------------------------------------------------------
user_input = st.chat_input("Nhập câu hỏi tài chính tại đây...")

if prompt_to_trigger:
    user_input = prompt_to_trigger

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        status_placeholder = st.empty()
        status_placeholder.markdown("⏳ *Đang định tuyến & truy xuất dữ liệu BCTC...*")

        t0 = time.time()
        question = user_input.strip()

        tickers = corpus.infer_tickers(question)

        try:
            solution, route_used = run_smart_pipeline(question)
            elapsed = time.time() - t0
            status_placeholder.empty()

            docs_list = list(solution.relevant_docs)
            evidence_list = []
            if solution.evidence:
                for ev in solution.evidence:
                    df_dict = ev.frame.to_dict(orient="records")
                    evidence_list.append({"variable": ev.variable, "frame": df_dict})

            res_payload = {
                "text_summary": f"Dựa trên phân tích báo cáo tài chính của **{', '.join(tickers) if tickers else 'doanh nghiệp niêm yết'}**:",
                "answer": solution.answer,
                "docs": docs_list,
                "evidence": evidence_list,
                "pandas_query": solution.pandas_query,
                "method": solution.method,
                "route": route_used,
                "elapsed": elapsed,
            }

            st.markdown(res_payload["text_summary"])
            st.markdown(f"""
            <div class="answer-card">
                <span style="color:#94a3b8;font-size:0.9rem;">KẾT QUẢ TÍNH TOÁN / TRA CỨU:</span><br>
                <span class="answer-value">{solution.answer:,.2f}</span>
            </div>
            """, unsafe_allow_html=True)

            if docs_list:
                st.markdown("**📄 Báo cáo tài chính liên quan:**")
                badges_html = "".join([f'<span class="metric-badge">📂 {doc}</span>' for doc in docs_list[:4]])
                st.markdown(badges_html, unsafe_allow_html=True)

            if solution.evidence:
                with st.expander("📋 Xem Bảng Bằng Chứng Dữ Liệu (Evidence Tables)", expanded=False):
                    for ev in solution.evidence:
                        st.caption(f"Biến: `{ev.variable}`")
                        st.dataframe(ev.frame, use_container_width=True)

            if solution.pandas_query:
                with st.expander("🔢 Mã lệnh Pandas tính toán", expanded=False):
                    st.code(solution.pandas_query, language="python")

            st.caption(f"⚡ Thời gian phản hồi: `{elapsed:.2f}s` | Tuyến xử lý: `{route_used}` | Phương pháp: `{solution.method}`")

            st.session_state.messages.append({"role": "assistant", "content": res_payload})

        except Exception as exc:
            status_placeholder.empty()
            error_msg = str(exc)
            error_content = f"❌ Không thể tìm thấy hoặc tính toán được đáp án cho câu hỏi này.\n\n*Chi tiết:* `{error_msg}`"
            st.error(error_content)
            st.session_state.messages.append({"role": "assistant", "content": error_content})
