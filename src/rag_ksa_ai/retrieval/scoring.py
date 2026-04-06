import re
import unicodedata

from rag_ksa_ai.config import STAFF_QUERY_TERMS


def keyword_score(query: str, text: str) -> float:
    words = [w for w in re.findall(r"[\u0600-\u06FF]+|\d+", query) if len(w) > 1]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in text)
    return hits / len(words)


def is_staff_query(query: str) -> bool:
    q = query.strip()
    return any(k in q for k in STAFF_QUERY_TERMS)


def _name_tokens(full_name: str) -> list:
    if not full_name:
        return []

    txt = str(full_name)
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"[\u064B-\u0652]", "", txt)
    txt = txt.lower()
    txt = re.sub(r"\b(د|دكتور|دكتورة|ا\.م\.د|أ\.م\.د|ا\.د|أ\.د)\b", " ", txt)
    txt = txt.replace("/", " ").replace(".", " ")

    stop_tokens = {
        "محمد", "احمد", "أحمد", "عبد", "ابو", "أبو", "بن", "ابن",
        "ال", "الشيخ", "سيد", "عيد", "علي", "حسن", "محمود",
    }
    tokens = [t for t in re.findall(r"[\u0600-\u06FFA-Za-z]+", txt) if len(t) >= 2]
    filtered = [t for t in tokens if t not in stop_tokens]
    return filtered or tokens


def staff_name_match_score(query: str, entry: dict) -> float:
    q = query.strip()
    name = entry.get("title_ar") or entry.get("full_name") or ""
    tokens = _name_tokens(name)
    query_tokens = _name_tokens(q)
    if not tokens:
        return 0.0

    matched = 0
    for t in tokens:
        if t in q or t in query_tokens:
            matched += 1

    if query_tokens:
        overlap = sum(1 for t in query_tokens if t in tokens)
        matched = max(matched, overlap)

    if matched == 0:
        return 0.0

    base = matched / max(1, len(tokens))
    if len(query_tokens) == 1 and query_tokens[0] in tokens:
        base = max(base, 0.45)

    return min(1.0, base)
