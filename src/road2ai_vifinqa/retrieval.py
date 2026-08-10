"""Entity-constrained lexical row and table retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from .corpus import Corpus, DocumentRef, RowAsset, TableAsset
from .text import fold_text


STOPWORDS = frozenset(
    "cua cong ty me co phan tap doan ngan hang tmcp tong trong vao nam cuoi den ngay "
    "la bao nhieu trieu ty nghin tram bao cao cac va cho mot".split()
)
YEAR_RE = re.compile(r"\b20\d{2}\b")


@dataclass(frozen=True, slots=True)
class RowHit:
    score: float
    row: RowAsset
    table: TableAsset
    document: DocumentRef


def metric_phrase(question: str, *, tickers: list[str] | None = None) -> str:
    folded = fold_text(question)
    folded = YEAR_RE.sub(" ", folded)
    entity_patterns = (
        r"\s+cua\s+cong\s+ty\s+me\b",
        r"\s+cua\s+(?:ctcp|ngan hang|tong cong ty|tap doan|cong ty co phan)\b",
        r"\s+tai\s+(?:cong ty me|ctcp|ngan hang|tong cong ty|tap doan)\b",
    )
    cut_positions = [m.start() for pattern in entity_patterns if (m := re.search(pattern, folded))]
    for ticker in tickers or []:
        match = re.search(rf"\s+cua\s+(?:cong\s+ty\s+me\s+)?{re.escape(ticker.casefold())}\b", folded)
        if match:
            cut_positions.append(match.start())
    if cut_positions:
        folded = folded[: min(cut_positions)]
    for separator in (" la bao nhieu", " bang bao nhieu"):
        if separator in folded:
            folded = folded.split(separator, 1)[0]
    folded = re.sub(r"\b(?:cuoi|dau|trong) nam\b", " ", folded)
    folded = re.sub(r"\b(?:bao nhieu )?(?:trieu|ty|nghin ty|tram ty) dong\b", " ", folded)
    return " ".join(token for token in folded.split() if token not in {"nam"} and not token.isdigit())


def _idf_weights(rows: list[RowAsset], query_tokens: set[str]) -> dict[str, float]:
    if not rows:
        return {token: 1.0 for token in query_tokens}
    counts: Counter[str] = Counter()
    for row in rows:
        present = set(row.folded_text.split()) & query_tokens
        counts.update(present)
    return {
        token: math.log((len(rows) + 1) / (counts[token] + 1)) + 1.0 for token in query_tokens
    }


def retrieve_rows(
    corpus: Corpus, question: str, *, limit: int = 20, include_prior: bool = False
) -> list[RowHit]:
    documents = corpus.documents_for_question(question, include_prior=include_prior)
    rows = corpus.rows_for_documents(documents)
    years = set(corpus.infer_years(question))
    folded_question = fold_text(question)
    phrase = metric_phrase(question, tickers=corpus.infer_tickers(question))
    qtokens = set(phrase.split())
    if not qtokens:
        qtokens = set(fold_text(question).split()) - STOPWORDS
    weights = _idf_weights(rows, qtokens)
    doc_by_id = {doc.doc_id: doc for doc in documents}
    table_cache: dict[tuple[str, int], TableAsset] = {}
    scored: list[RowHit] = []

    for row in rows:
        rtokens = set(row.folded_text.split())
        overlap = qtokens & rtokens
        if not overlap:
            continue
        weighted_recall = sum(weights[token] for token in overlap) / max(
            sum(weights.values()), 1e-9
        )
        precision = len(overlap) / max(len(rtokens & (qtokens | STOPWORDS)), 1)
        sequence = SequenceMatcher(None, phrase, row.folded_text).ratio()
        exact = 1.0 if phrase and phrase in row.folded_text else 0.0
        short_bonus = 1.0 / (1.0 + max(0, len(rtokens) - len(qtokens)) / 8.0)
        score = 7.0 * weighted_recall + 1.5 * precision + 2.5 * sequence + 5.0 * exact + short_bonus
        key = (row.doc_id, row.table_id)
        if key not in table_cache:
            table_cache[key] = corpus.table(*key)
        table = table_cache[key]
        context_tokens = set(fold_text(table.context).split())
        folded_context = fold_text(table.context)
        score += 2.2 * len(context_tokens & qtokens) / max(len(qtokens), 1)
        if any(marker in folded_question for marker in ("cuoi nam", "den ngay", "vao ngay")):
            if any(marker in folded_context for marker in ("bang can doi", "bao cao tinh hinh tai chinh")):
                score += 2.4
        if "nganh " in phrase:
            subtype = phrase.split("nganh ", 1)[1].strip()
            if subtype and subtype in row.folded_text:
                score += 6.0
        if hit_year := doc_by_id[row.doc_id].report_year:
            score += 1.5 if hit_year in years else (-0.5 if years else 0.0)
        scored.append(RowHit(score, row, table, doc_by_id[row.doc_id]))

    scored.sort(key=lambda hit: (-hit.score, hit.document.report_year, hit.row.table_id, hit.row.row_idx))
    return scored[:limit]
