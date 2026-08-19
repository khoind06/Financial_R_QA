"""
Financial Report QA - Smart Conversational AI Assistant (Full Pipeline Demo)
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
    .stApp { max-width: 1200px; margin: 0 auto; }
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
        font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .chat-header p { color: #94a3b8; font-size: 0.95rem; }
    .metric-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; background: #1e293b; color: #38bdf8;
        border: 1px solid #334155; margin-right: 6px; margin-bottom: 6px;
    }
    .answer-card {
        background: #0f172a; border: 1px solid #1e293b;
        border-left: 5px solid #10b981; padding: 1.2rem;
        border-radius: 12px; margin: 0.8rem 0;
    }
    .answer-value { font-size: 1.8rem; font-weight: 700; color: #10b981; }
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

    llm_mode = st.radio("Động cơ LLM", ["Cloud API (Groq/Gemini)", "Local Ollama"], index=0)

    if llm_mode == "Cloud API (Groq/Gemini)":
        api_key = st.text_input("API Key", type="password",
                                placeholder="Dán API Key (gsk_... hoặc AIzaSy...)",
                                help="Lấy API Key FREE tại console.groq.com hoặc aistudio.google.com")
        provider = st.selectbox("Provider", ["Groq Cloud", "Google Gemini", "OpenAI / Khác"])
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
# Smart Query Handler — covers every category a teacher/evaluator may ask
# ---------------------------------------------------------------------------

SAFE_DIV = lambda n, d: n / d if d and not math.isnan(d) and d != 0 else math.nan  # noqa: E731
BILLION = 1_000_000_000.0
MILLION = 1_000_000.0

# ------ Metric Config: maps keywords → (panel column, display name, default_scale) ------
METRIC_MAP = [
    # Revenue / Profit
    (["doanh thu thuan", "doanh thu"],           "net_revenue",          "Doanh thu thuần (tỷ đồng)",      BILLION),
    (["gia von hang ban", "gia von"],             "cogs",                 "Giá vốn hàng bán (tỷ đồng)",     BILLION),
    (["loi nhuan gop"],                           "gross_profit",         "Lợi nhuận gộp (tỷ đồng)",        BILLION),
    (["loi nhuan truoc thue", "loi nhuan truoc"], "pbt",                  "Lợi nhuận trước thuế (tỷ đồng)", BILLION),
    (["loi nhuan sau thue", "loi nhuan rong", "loi nhuan"],  "npat",     "Lợi nhuận sau thuế (tỷ đồng)",   BILLION),
    (["loi nhuan hoat dong", "loi tu hoat dong"], "operating_profit",     "Lợi nhuận từ HĐKD (tỷ đồng)",   BILLION),
    (["chi phi ban hang"],                        "selling_expense",      "Chi phí bán hàng (tỷ đồng)",     BILLION),
    (["chi phi quan ly", "quan ly doanh nghiep"], "admin_expense",        "Chi phí quản lý DN (tỷ đồng)",   BILLION),
    (["chi phi lai vay", "lai vay"],              "interest_expense",     "Chi phí lãi vay (tỷ đồng)",      BILLION),
    # Balance sheet
    (["tong tai san"],                            "total_assets",         "Tổng tài sản (tỷ đồng)",         BILLION),
    (["tai san ngan han", "tai san co dinh ngan han"], "current_assets",  "Tài sản ngắn hạn (tỷ đồng)",     BILLION),
    (["tai san dai han"],                         "long_term_assets",     "Tài sản dài hạn (tỷ đồng)",      BILLION),
    (["tien va cac khoan tuong duong tien", "tien mat", "tien"],  "cash", "Tiền & tương đương tiền (tỷ)",   BILLION),
    (["hang ton kho", "ton kho"],                 "inventory",            "Hàng tồn kho (tỷ đồng)",         BILLION),
    (["no phai tra", "tong no"],                  "liabilities",          "Nợ phải trả (tỷ đồng)",          BILLION),
    (["no ngan han"],                             "current_liabilities",  "Nợ ngắn hạn (tỷ đồng)",          BILLION),
    (["von chu so huu", "von chu"],               "equity",               "Vốn chủ sở hữu (tỷ đồng)",       BILLION),
    # Cash flow
    (["luu chuyen tien thuan", "dong tien hoat dong", "luong tien"],  "cfo", "Dòng tiền từ HĐKD (tỷ đồng)", BILLION),
    # Ratios
    (["bien loi nhuan gop", "ty le loi nhuan gop"],  "gross_margin",     "Biên lợi nhuận gộp (%)",          1.0),
    (["bien loi nhuan rong", "ty le loi nhuan rong"], "net_margin",      "Biên lợi nhuận ròng (%)",          1.0),
    (["bien loi nhuan hoat dong"],                "operating_margin",     "Biên lợi nhuận HĐKD (%)",         1.0),
    (["roe", "ty suat sinh loi tren von chu"],    "roe",                  "ROE (%)",                         1.0),
    (["roa", "ty suat sinh loi tren tai san"],    "roa",                  "ROA (%)",                         1.0),
    (["he so no tren von", "no tren von", "d/e"], "liabilities_to_equity","Nợ/Vốn CSH (lần)",               1.0),
    (["he so no tren tai san", "no tren tai san"],"liabilities_to_assets","Nợ/Tổng TS (lần)",               1.0),
    (["he so thanh toan hien hanh", "kha nang thanh toan ngan han", "thanh toan hien hanh"], "current_ratio","Hệ số thanh toán hiện hành", 1.0),
    (["he so thanh toan nhanh", "thanh toan nhanh"], "quick_ratio",      "Hệ số thanh toán nhanh",          1.0),
    (["vong quay tai san", "vong quay tong tai san"], "asset_turnover",   "Vòng quay tổng tài sản",          1.0),
    (["vong quay hang ton kho", "so ngay ton kho"], "inventory_days",    "Số ngày tồn kho (ngày)",           1.0),
    (["tang truong doanh thu", "muc tang truong doanh thu"], "revenue_growth", "Tăng trưởng doanh thu (%)", 1.0),
]

YEAR_RE = re.compile(r"\b(20\d{2})\b")

def _detect_metric(folded: str) -> tuple[str, str, float] | None:
    """Return (panel_column, display_name, scale) for the first matching keyword."""
    for keywords, col, name, scale in METRIC_MAP:
        if any(k in folded for k in keywords):
            return col, name, scale
    return None


def _format_val(val: float, scale: float, name: str) -> str:
    if math.isnan(val):
        return "N/A"
    if scale == 1.0:
        return f"{val:.2f}"
    return f"{val / scale:,.2f}"


def _panel_rows(ticker: str, years: list[int]) -> pd.DataFrame:
    df = panel.frame
    mask = df["ticker"] == ticker
    if years:
        mask &= df["year"].isin(years)
    return df[mask].copy()


# ── 1. Single-ticker, single-metric, single-year lookup ──────────────────────
def try_single_lookup(question: str, folded: str):
    """Handle: 'Doanh thu thuần của VNM năm 2023 là bao nhiêu?'"""
    from road2ai_vifinqa.submission import SubmissionSolution, EvidenceFrame

    tickers = corpus.infer_tickers(question)
    years   = corpus.infer_years(question)
    metric  = _detect_metric(folded)

    if not tickers or not years or metric is None:
        return None

    ticker      = tickers[0]
    year        = years[0]
    col, name, scale = metric

    rows = _panel_rows(ticker, [year])
    if rows.empty or col not in rows.columns or math.isnan(rows.iloc[0][col]):
        return None

    val     = float(rows.iloc[0][col])
    disp    = _format_val(val, scale, name)
    # answer in the panel native unit (VND or ratio)
    answer  = val if scale == 1.0 else val / scale

    # Build evidence frame
    cname   = corpus.company_names.get(ticker, ticker)
    ev_df   = rows[["ticker", "year", col]].rename(columns={col: name})
    ev_df.insert(1, "Tên công ty", cname)

    # Try to get panel cell for doc/table ref
    raw_col_map = {v: k for k, v in __import__(
        "road2ai_vifinqa.panel", fromlist=["RAW_COLUMNS"]).RAW_COLUMNS.items()}
    cell     = panel.cell(ticker, year, raw_col_map.get(col, col))
    docs     = [cell.doc_id] if cell else [f"{ticker}_financial_statements_{year}_consolidated"]
    tables   = [cell.table_ref] if cell else []

    pq = (
        f"df[(df.ticker=='{ticker}') & (df.year=={year})]['{col}'].iloc[0]"
        if scale == 1.0 else
        f"df[(df.ticker=='{ticker}') & (df.year=={year})]['{col}'].iloc[0] / {scale:.0f}"
    )
    return SubmissionSolution(
        id=9999, question=question, answer=answer,
        relevant_docs=tuple(docs), relevant_tables=tuple(tables),
        evidence=(EvidenceFrame(variable=f"{ticker}_{year}_{col}", frame=ev_df),),
        pandas_query=pq,
        method=f"panel_single_lookup:{col}",
        confidence=0.99,
    ), "panel_lookup"


# ── 2. Multi-year trend for single ticker ─────────────────────────────────────
def try_multi_year_trend(question: str, folded: str):
    """Handle: 'Doanh thu của VNM từ 2019 đến 2023' or 'So sánh ROE HPG qua các năm'"""
    from road2ai_vifinqa.submission import SubmissionSolution, EvidenceFrame

    trend_kw = ["qua cac nam", "tu nam", "den nam", "giai doan", "so sanh qua", "bien dong"]
    if not any(k in folded for k in trend_kw):
        return None

    tickers = corpus.infer_tickers(question)
    metric  = _detect_metric(folded)
    if not tickers or metric is None:
        return None

    ticker      = tickers[0]
    col, name, scale = metric
    years_found  = sorted(corpus.infer_years(question))

    df = panel.frame[panel.frame["ticker"] == ticker].copy()
    if years_found:
        df = df[df["year"].between(min(years_found), max(years_found))]

    if df.empty or col not in df.columns:
        return None

    cname   = corpus.company_names.get(ticker, ticker)
    ev_df   = df[["year", col]].copy()
    ev_df[name] = (ev_df[col] / scale).round(2) if scale != 1.0 else ev_df[col].round(2)
    ev_df   = ev_df[["year", name]].rename(columns={"year": "Năm"}).reset_index(drop=True)

    # Latest year value as scalar answer
    latest  = df.sort_values("year").iloc[-1][col]
    answer  = latest if scale == 1.0 else latest / scale

    pq = (
        f"df[df.ticker=='{ticker}'][['year','{col}']].sort_values('year')"
        if scale == 1.0 else
        f"df[df.ticker=='{ticker}'][['year','{col}']].assign(val=lambda r: r['{col}']/{scale:.0f}).sort_values('year')"
    )
    return SubmissionSolution(
        id=9999, question=question, answer=float(answer),
        relevant_docs=(f"{ticker}_financial_statements_consolidated",),
        relevant_tables=(),
        evidence=(EvidenceFrame(variable=f"{ticker}_trend_{col}", frame=ev_df),),
        pandas_query=pq,
        method=f"panel_trend:{col}",
        confidence=0.97,
    ), "trend"


# ── 3. Compare two tickers on a metric ───────────────────────────────────────
def try_compare_two_tickers(question: str, folded: str):
    """Handle: 'So sánh ROE của VNM và HPG năm 2023'"""
    from road2ai_vifinqa.submission import SubmissionSolution, EvidenceFrame

    compare_kw = ["so sanh", "cao hon", "thap hon", "lon hon", "nho hon", "hay", "va", "giua"]
    if not any(k in folded for k in compare_kw):
        return None

    tickers = corpus.infer_tickers(question)
    years   = corpus.infer_years(question)
    metric  = _detect_metric(folded)

    if len(tickers) < 2 or metric is None:
        return None

    col, name, scale = metric
    year = years[0] if years else 2023

    rows = []
    for t in tickers[:3]:
        df_t = _panel_rows(t, [year])
        if df_t.empty or col not in df_t.columns:
            continue
        val = float(df_t.iloc[0][col])
        cname = corpus.company_names.get(t, t)
        rows.append({"Mã CK": t, "Tên Công Ty": cname, name: round(val / scale, 2) if scale != 1.0 else round(val, 2)})

    if not rows:
        return None

    ev_df   = pd.DataFrame(rows)
    best    = max(rows, key=lambda r: r[name])
    answer  = best[name]

    pq = (
        f"df[df.ticker.isin({tickers[:3]}) & (df.year=={year})][['ticker','{col}']].sort_values('{col}', ascending=False)"
    )
    return SubmissionSolution(
        id=9999, question=question, answer=float(answer),
        relevant_docs=tuple(f"{t}_financial_statements_{year}_consolidated" for t in tickers[:3]),
        relevant_tables=(),
        evidence=(EvidenceFrame(variable=f"compare_{col}_{year}", frame=ev_df),),
        pandas_query=pq,
        method=f"panel_compare:{col}",
        confidence=0.97,
    ), "compare"


# ── 4. Top-N ranking across all 100 tickers ───────────────────────────────────
def try_top_ranking_query(question: str, folded: str):
    """Handle: 'Top 10 công ty doanh thu cao nhất năm 2023'"""
    from road2ai_vifinqa.submission import SubmissionSolution, EvidenceFrame

    ranking_kw = ["top", "xep hang", "bang xep hang", "danh sach", "cao nhat", "thap nhat",
                  "nhieu nhat", "it nhat", "lon nhat", "nho nhat"]
    if not any(k in folded for k in ranking_kw):
        return None

    years  = corpus.infer_years(question)
    year   = str(years[0]) if years else "2023"
    metric = _detect_metric(folded)
    if metric is None:
        metric = ("net_revenue", "Doanh thu thuần (tỷ đồng)", BILLION)

    col, name, scale = metric

    # Determine N
    n_match = re.search(r"\b(\d+)\b", question)
    n = int(n_match.group(1)) if n_match and int(n_match.group(1)) <= 100 else 10
    ascending = any(k in folded for k in ["thap nhat", "it nhat", "nho nhat"])

    records = []
    for ticker, years_data in panel.raw.items():
        from road2ai_vifinqa.panel import RAW_COLUMNS
        raw_key_map = {v: k for k, v in RAW_COLUMNS.items()}
        raw_col = raw_key_map.get(col)
        if raw_col is None:
            # ratio column — pull from enriched frame
            df_t = panel.frame[(panel.frame["ticker"] == ticker) & (panel.frame["year"] == int(year))]
            if df_t.empty or col not in df_t.columns:
                continue
            val = float(df_t.iloc[0][col])
        else:
            cell_key = RAW_COLUMNS.get(raw_col, raw_col)
            cell = (panel.raw.get(ticker, {}).get(year) or {}).get(cell_key)
            if cell is None:
                continue
            val = float(cell["value"])

        disp_val = val / scale if scale != 1.0 else val
        cname    = corpus.company_names.get(ticker, ticker)
        records.append({"Mã CK": ticker, "Tên Doanh Nghiệp": cname, name: round(disp_val, 2)})

    if not records:
        return None

    records.sort(key=lambda x: x[name], reverse=not ascending)
    df_rank = pd.DataFrame(records[:n])
    df_rank.index = range(1, len(df_rank) + 1)
    df_rank.index.name = "Thứ Hạng"

    top_val = float(records[0][name])
    docs    = [f"{r['Mã CK']}_financial_statements_{year}_consolidated" for r in records[:5]]

    pq = (
        f"panel[panel.year=={year}][['ticker','{col}']].sort_values('{col}', ascending={ascending}).head({n})"
    )
    return SubmissionSolution(
        id=9999, question=question, answer=top_val,
        relevant_docs=tuple(docs), relevant_tables=(),
        evidence=(EvidenceFrame(variable=f"top{n}_{year}_{col}", frame=df_rank),),
        pandas_query=pq,
        method=f"panel_top{n}_ranking:{col}",
        confidence=0.99,
    ), "ranking"


# ── 5. Full financial summary profile for a ticker+year ──────────────────────
def try_company_profile(question: str, folded: str):
    """Handle: 'Phân tích tổng quan tài chính của HPG năm 2023' or 'Hồ sơ tài chính VNM 2022'"""
    from road2ai_vifinqa.submission import SubmissionSolution, EvidenceFrame

    profile_kw = ["tong quan", "ho so", "phan tich tai chinh", "buc tranh tai chinh",
                  "tinh hinh tai chinh", "ket qua kinh doanh tong the"]
    if not any(k in folded for k in profile_kw):
        return None

    tickers = corpus.infer_tickers(question)
    years   = corpus.infer_years(question)
    if not tickers:
        return None

    ticker = tickers[0]
    year   = years[0] if years else 2023
    rows   = _panel_rows(ticker, [year])
    if rows.empty:
        return None

    row     = rows.iloc[0]
    cname   = corpus.company_names.get(ticker, ticker)
    summary = []
    for _, name, scale in [(k, n, s) for k, n, s in [(c, n, s) for _, c, n, s in METRIC_MAP]]:
        col2 = next((c for _, c, n2, _ in METRIC_MAP if n2 == name), None)
        if col2 and col2 in row and not math.isnan(row[col2]):
            summary.append({"Chỉ tiêu": name, "Giá trị": round(float(row[col2]) / scale if scale != 1.0 else float(row[col2]), 2)})

    if not summary:
        return None

    ev_df   = pd.DataFrame(summary)
    val_npat = float(row["npat"]) / BILLION if "npat" in row and not math.isnan(row["npat"]) else math.nan

    pq = f"df[(df.ticker=='{ticker}') & (df.year=={year})].T"
    return SubmissionSolution(
        id=9999, question=question, answer=round(val_npat, 2) if not math.isnan(val_npat) else 0.0,
        relevant_docs=(f"{ticker}_financial_statements_{year}_consolidated",),
        relevant_tables=(),
        evidence=(EvidenceFrame(variable=f"{ticker}_{year}_profile", frame=ev_df),),
        pandas_query=pq,
        method=f"panel_company_profile:{ticker}:{year}",
        confidence=0.97,
    ), "profile"


# ── 6. Year-over-year growth calculation ──────────────────────────────────────
def try_yoy_growth(question: str, folded: str):
    """Handle: 'Doanh thu VNM 2023 tăng bao nhiêu % so với 2022?'"""
    from road2ai_vifinqa.submission import SubmissionSolution, EvidenceFrame

    growth_kw = ["tang bao nhieu", "giam bao nhieu", "tang truong", "so voi nam truoc",
                 "bien dong", "thay doi", "so voi", "tang hay giam"]
    if not any(k in folded for k in growth_kw):
        return None

    tickers = corpus.infer_tickers(question)
    years   = sorted(corpus.infer_years(question))
    metric  = _detect_metric(folded)
    if not tickers or metric is None:
        return None

    ticker      = tickers[0]
    col, name, scale = metric

    if len(years) >= 2:
        yr_curr, yr_prev = years[-1], years[-2]
    else:
        yr_curr = years[0] if years else 2023
        yr_prev = yr_curr - 1

    r_curr = _panel_rows(ticker, [yr_curr])
    r_prev = _panel_rows(ticker, [yr_prev])
    if r_curr.empty or r_prev.empty or col not in r_curr.columns:
        return None

    v_curr = float(r_curr.iloc[0][col])
    v_prev = float(r_prev.iloc[0][col])
    if math.isnan(v_curr) or math.isnan(v_prev) or v_prev == 0:
        return None

    growth  = (v_curr - v_prev) / abs(v_prev) * 100
    cname   = corpus.company_names.get(ticker, ticker)
    ev_df   = pd.DataFrame([
        {"Năm": yr_prev, name: round(v_prev / scale, 2) if scale != 1.0 else round(v_prev, 2)},
        {"Năm": yr_curr, name: round(v_curr / scale, 2) if scale != 1.0 else round(v_curr, 2)},
        {"Năm": "Tăng trưởng %", name: round(growth, 2)},
    ])

    pq = (
        f"((df[(df.ticker=='{ticker}') & (df.year=={yr_curr})]['{col}'].iloc[0] - "
        f"df[(df.ticker=='{ticker}') & (df.year=={yr_prev})]['{col}'].iloc[0]) / "
        f"abs(df[(df.ticker=='{ticker}') & (df.year=={yr_prev})]['{col}'].iloc[0])) * 100"
    )
    return SubmissionSolution(
        id=9999, question=question, answer=round(growth, 2),
        relevant_docs=(
            f"{ticker}_financial_statements_{yr_curr}_consolidated",
            f"{ticker}_financial_statements_{yr_prev}_consolidated",
        ),
        relevant_tables=(),
        evidence=(EvidenceFrame(variable=f"{ticker}_yoy_{col}", frame=ev_df),),
        pandas_query=pq,
        method=f"panel_yoy_growth:{col}",
        confidence=0.98,
    ), "yoy_growth"


# ── Master Smart Dispatcher ───────────────────────────────────────────────────
def run_smart_pipeline(question: str):
    from road2ai_vifinqa.solve import route_for_id
    from road2ai_vifinqa.pipeline import (
        solve_easy_submission, solve_hard_submission,
        solve_note_submission, solve_template_submission,
    )

    from road2ai_vifinqa.text import fold_text
    q_clean = question.strip()
    folded  = fold_text(q_clean)

    # ── Phase 0: Panel-based custom handlers (no LLM needed, instant) ─────────
    for handler in [
        try_yoy_growth,
        try_multi_year_trend,
        try_compare_two_tickers,
        try_top_ranking_query,
        try_company_profile,
        try_single_lookup,
    ]:
        result = handler(q_clean, folded)
        if result is not None:
            return result

    # ── Phase 1: Known dataset question → exact route ──────────────────────────
    qid = question_by_text.get(q_clean, 9999)
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

    # ── Phase 2: Custom question → LLM-assisted cascade ───────────────────────
    for solver, route_name in [
        (lambda: solve_template_submission(9999, q_clean, template_solver), "template"),
        (lambda: solve_hard_submission(9999, q_clean, panel), "hard"),
        (lambda: solve_note_submission(9999, q_clean, corpus, max_attempts=3, log_path=None), "note"),
    ]:
        try:
            sol = solver()
            return sol, route_name
        except Exception:
            pass

    # ── Phase 3: Final fallback to LLM Easy Solver ────────────────────────────
    sol = solve_easy_submission(9999, q_clean, corpus, max_attempts=3, log_path=None)
    return sol, "direct"


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

# Render Quick Action Chips
st.markdown("**💡 Thử hỏi các câu hỏi mẫu:**")
chip_cols = st.columns(4)
suggested_queries = [
    "Doanh thu thuần của FPT năm 2023 là bao nhiêu tỷ đồng?",
    "So sánh ROE của VNM và HPG năm 2022",
    "Top 10 công ty có lợi nhuận cao nhất năm 2023",
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

# Render Chat History
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
                        st.dataframe(pd.DataFrame(ev["frame"]), use_container_width=True)
            if data.get("pandas_query"):
                with st.expander("🔢 Mã lệnh Pandas tính toán", expanded=False):
                    st.code(data["pandas_query"], language="python")
            if data.get("elapsed"):
                st.caption(f"⚡ `{data['elapsed']:.2f}s` | Tuyến: `{data.get('route', 'Auto')}` | Phương pháp: `{data.get('method', '-')}`")

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
        status_ph = st.empty()
        status_ph.markdown("⏳ *Đang định tuyến & truy xuất dữ liệu BCTC...*")

        t0       = time.time()
        question = user_input.strip()
        tickers  = corpus.infer_tickers(question)

        try:
            solution, route_used = run_smart_pipeline(question)
            elapsed = time.time() - t0
            status_ph.empty()

            docs_list = list(solution.relevant_docs)
            evidence_frames = []
            if solution.evidence:
                for ev in solution.evidence:
                    evidence_frames.append({"variable": ev.variable, "frame": ev.frame.to_dict(orient="records")})

            cnames = [corpus.company_names.get(t, t) for t in (tickers or [])]
            summary_label = ", ".join(cnames) if cnames else "dữ liệu tài chính"
            res_payload = {
                "text_summary": f"Dựa trên phân tích BCTC của **{summary_label}**:",
                "answer": solution.answer,
                "docs": docs_list,
                "evidence": evidence_frames,
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
                badges_html = "".join([f'<span class="metric-badge">📂 {d}</span>' for d in docs_list[:4]])
                st.markdown(badges_html, unsafe_allow_html=True)

            if solution.evidence:
                expanded = route_used in ("ranking", "compare", "profile", "trend")
                with st.expander("📋 Xem Bảng Bằng Chứng Dữ Liệu (Evidence Tables)", expanded=expanded):
                    for ev in solution.evidence:
                        st.caption(f"Biến: `{ev.variable}`")
                        st.dataframe(ev.frame, use_container_width=True)

            if solution.pandas_query:
                with st.expander("🔢 Mã lệnh Pandas tính toán", expanded=False):
                    st.code(solution.pandas_query, language="python")

            st.caption(f"⚡ `{elapsed:.2f}s` | Tuyến: `{route_used}` | Phương pháp: `{solution.method}`")
            st.session_state.messages.append({"role": "assistant", "content": res_payload})

        except Exception as exc:
            status_ph.empty()
            err = str(exc)
            error_content = f"❌ Không thể tìm thấy hoặc tính toán được đáp án cho câu hỏi này.\n\n*Chi tiết:* `{err}`"
            st.error(error_content)
            st.session_state.messages.append({"role": "assistant", "content": error_content})
