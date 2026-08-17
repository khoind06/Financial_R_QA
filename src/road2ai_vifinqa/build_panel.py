"""Extract the canonical statement cube used by formula questions."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time

from .paths import INDEX_PATH, PANEL_MANIFEST_PATH, PANEL_PATH
from .text import fold_text, parse_vn_number, source_scale


CODE_RE = re.compile(r"^\d{1,3}$")
KQKD_REQUIRED = frozenset({"01", "11", "20", "25", "26"})
COST_KEYS = frozenset(f"kqkd:{code}" for code in ("11", "22", "23", "25", "26", "32", "51", "52"))
KNOWN_CODES = {
    "kqkd": frozenset({"01", "02", "10", "11", "20", "21", "22", "23", "25", "26", "30", "31", "32", "40", "50", "51", "52", "60"}),
    "cdkt": frozenset({"100", "110", "120", "130", "140", "150", "200", "210", "220", "230", "240", "250", "260", "270", "300", "310", "320", "330", "400", "410", "420", "430", "440"}),
    "lctt": frozenset({f"{value:02d}" for value in range(1, 81)}),
}


def _find_code(row: list[str]) -> tuple[int, str] | None:
    for idx, raw in enumerate(row[:3]):
        value = raw.strip()
        if CODE_RE.fullmatch(value):
            return idx, value
    return None

LABEL_KEY_MAP = {
    "kqkd": [
        ("loi nhuan sau thue tndn", "60"),
        ("loi nhuan sau thue", "60"),
        ("tong loi nhuan ke toan truoc thue", "50"),
        ("loi nhuan truoc thue", "50"),
        ("doanh thu thuan ve ban hang", "10"),
        ("doanh thu thuan", "10"),
        ("loi nhuan gop", "20"),
        ("gia von hang ban", "11"),
        ("chi phi tai chinh", "22"),
        ("chi phi lai vay", "23"),
        ("chi phi ban hang", "25"),
        ("chi phi quan ly", "26"),
        ("doanh thu ban hang", "01"),
    ],
    "cdkt": [
        ("tong cong tai san", "270"),
        ("tong tai san", "270"),
        ("tong cong nguon von", "440"),
        ("von chu so huu", "400"),
        ("no phai tra", "300"),
        ("tai san ngan han", "100"),
        ("tai san dai han", "200"),
        ("tien va cac khoan tương duong tien", "110"),
        ("hang ton kho", "140"),
        ("no ngan han", "310"),
        ("no dai han", "330"),
    ],
}


def _infer_code_from_label(row: list[str], kind: str) -> tuple[int, str] | None:
    if kind not in LABEL_KEY_MAP:
        return None
    row_text = fold_text(" ".join(row[:2]))
    for phrase, code in LABEL_KEY_MAP[kind]:
        if phrase in row_text:
            return 0, code
    return None


def _label(row: list[str], code_idx: int) -> str:
    candidates: list[str] = []
    for idx, cell in enumerate(row):
        if idx == code_idx or parse_vn_number(cell) is not None:
            continue
        if cell.strip():
            candidates.append(cell.strip())
    return max(candidates, key=len, default="")


def _classify(rows: list[list[str]]) -> str | None:
    codes: set[str] = set()
    labels: list[str] = []
    for row in rows:
        found = _find_code(row)
        if found is None:
            continue
        idx, code = found
        codes.add(code)
        if len(code) == 1:
            codes.add(code.zfill(2))
        labels.append(fold_text(_label(row, idx)))
    if len({code for code in codes if len(code) == 3}) >= 5:
        return "cdkt"
    if any("chuyen tien" in label for label in labels):
        return "lctt"
    if len(codes & KQKD_REQUIRED) >= 3:
        return "kqkd"
    return None


def _numeric_values(row: list[str], code_idx: int) -> list[tuple[int, float, str]]:
    values: list[tuple[int, float, str]] = []
    for idx in range(code_idx + 1, len(row)):
        raw = row[idx].strip()
        value = parse_vn_number(raw)
        if value is None:
            continue
        compact = raw.replace(".", "").replace(",", "").replace(" ", "")
        unsigned = compact.strip("+-()")
        if unsigned.isdigit() and len(unsigned) <= 3:
            continue
        values.append((idx, value, raw))
    return values


def build_panel(*, force: bool = False) -> dict[str, object]:
    if PANEL_PATH.exists() and PANEL_MANIFEST_PATH.exists() and not force:
        return json.loads(PANEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    started = time.time()
    conn = sqlite3.connect(f"file:{INDEX_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    panel: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    table_counts = {"kqkd": 0, "cdkt": 0, "lctt": 0}
    duplicate_count = 0
    query = """
        SELECT d.ticker, d.report_year, d.scope, t.doc_id, t.table_id,
               t.context, t.rows_json
        FROM tables t JOIN documents d ON d.doc_id=t.doc_id
        WHERE d.scope != 'parent'
        ORDER BY d.ticker, d.report_year,
                 CASE d.scope WHEN 'consolidated' THEN 0 ELSE 1 END,
                 t.doc_id, t.table_id
    """
    try:
        for item in conn.execute(query):
            rows: list[list[str]] = json.loads(item["rows_json"])
            kind = _classify(rows)
            if kind is None:
                continue
            table_counts[kind] += 1
            header = " ".join(" ".join(row) for row in rows[: min(5, len(rows))])
            scale = source_scale(f"{item['context']} {header}")
            ticker = item["ticker"]
            year = str(item["report_year"])
            bucket = panel.setdefault(ticker, {}).setdefault(year, {})
            for row_idx, row in enumerate(rows):
                found = _find_code(row) or _infer_code_from_label(row, kind)
                if found is None:
                    continue
                code_idx, code = found
                if kind != "cdkt":
                    code = code.zfill(2)
                if code not in KNOWN_CODES[kind]:
                    continue
                values = _numeric_values(row, code_idx)
                if not values:
                    continue
                col_idx, parsed, raw = values[0]
                key = f"{kind}:{code}"
                if key in bucket:
                    duplicate_count += 1
                    continue
                value = parsed * scale
                if key in COST_KEYS:
                    value = abs(value)
                bucket[key] = {
                    "value": value,
                    "raw": raw,
                    "label": _label(row, code_idx),
                    "doc_id": item["doc_id"],
                    "table_id": item["table_id"],
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "scale": scale,
                }
    finally:
        conn.close()

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PANEL_PATH.write_text(json.dumps(panel, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    cells = sum(len(metrics) for years in panel.values() for metrics in years.values())
    manifest: dict[str, object] = {
        "format_version": 1,
        "tickers": len(panel),
        "ticker_years": sum(len(years) for years in panel.values()),
        "cells": cells,
        "statement_tables": table_counts,
        "duplicates_kept_first": duplicate_count,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    PANEL_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_panel(force=args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
