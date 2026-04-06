import html
from flask import Blueprint, current_app, redirect, render_template, request, url_for

from rag_ksa_ai import (
    OLLAMA_MODEL,
    check_ollama,
    compose_staff_answer,
    format_retrieved_answer,
    generate_answer,
    is_staff_query,
)

bp = Blueprint("main", __name__)


def answer_to_html(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = []
    prev_key = None
    for ln in raw_lines:
        key = reformat_compare_key(ln)
        if key == prev_key:
            continue
        lines.append(ln)
        prev_key = key

    if not lines:
        return ""

    has_list = any(ln.startswith("-") or re_match_numbered_item(ln) for ln in lines)
    heading = None
    if has_list and lines and lines[0].endswith(":") and not lines[0].startswith("-"):
        heading = lines[0]
        lines = lines[1:]

    if has_list:
        html_parts = []
        if heading:
            html_parts.append(f"<h4 class=\"answer-subtitle\">{html.escape(heading.rstrip(':'))}</h4>")
        opened = False
        for ln in lines:
            if ln.startswith("-") or re_match_numbered_item(ln):
                if not opened:
                    html_parts.append("<ul>")
                    opened = True
                cleaned = ln.lstrip("- ")
                cleaned = remove_numbering_prefix(cleaned)
                html_parts.append(f"<li>{html.escape(cleaned)}</li>")
            else:
                if opened:
                    html_parts.append("</ul>")
                    opened = False
                html_parts.append(f"<p>{html.escape(ln)}</p>")
        if opened:
            html_parts.append("</ul>")
        return "\n".join(html_parts)

    paragraphs = [f"<p>{html.escape(ln)}</p>" for ln in lines]
    return "\n".join(paragraphs)


def reformat_compare_key(line: str) -> str:
    return " ".join(line.replace(":", "").replace("-", " ").split()).strip().lower()


def re_match_numbered_item(line: str) -> bool:
    if not line:
        return False
    if line[0].isdigit() and len(line) > 1 and line[1] in ".)":
        return True
    return False


def remove_numbering_prefix(line: str) -> str:
    if len(line) > 1 and line[0].isdigit() and line[1] in ".)":
        return line[2:].strip()
    return line


def source_to_html(text: str, max_lines: int = 5) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:max_lines]
    if not lines:
        return ""

    has_list = any(ln.startswith("-") or re_match_numbered_item(ln) for ln in lines)
    if has_list:
        items = []
        for ln in lines:
            if ln.startswith("-") or re_match_numbered_item(ln):
                cleaned = remove_numbering_prefix(ln.lstrip("- "))
                items.append(f"<li>{html.escape(cleaned)}</li>")
            else:
                items.append(f"<li>{html.escape(ln)}</li>")
        return "<ul class=\"source-list\">" + "".join(items) + "</ul>"

    joined = " ".join(lines)
    return f"<p>{html.escape(joined)}</p>"


def infer_query_intent(query: str) -> str:
    q = (query or "").strip()
    if any(k in q for k in ["كم", "عدد", "إحصائيات", "احصائيات", "إجمالي", "اجمالي"]):
        return "statistics"
    if any(k in q for k in ["مواد", "مقررات", "خطة", "المستوى", "الفصل"]):
        return "courses"
    if is_staff_query(q):
        return "staff"
    return "general"


def source_reason(src: dict, intent: str) -> str:
    src_type = src.get("type")
    title = src.get("title") or src.get("title_ar") or ""

    if intent == "statistics":
        if src_type == "statistics":
            return "تم اختيار المصدر لأنه يحتوي أرقام وإحصائيات مباشرة مرتبطة بسؤالك."
        if src_type in ("department", "departments"):
            return "تم اختيار المصدر لأنه يوضح توزيع الأعضاء على الأقسام."
    if intent == "staff":
        if src_type == "staff":
            return "تم اختيار المصدر لأنه ملف شخصي مباشر لعضو هيئة التدريس المطلوب."
    if intent == "courses" and src_type == "courses":
        return "تم اختيار المصدر لأنه يتضمن مقررات المستوى/الفصل المطلوب."

    if "إحصائ" in title or "إحصائ" in (src.get("text_ar") or ""):
        return "تم اختياره لوجود بيانات رقمية تدعم الإجابة."
    return "تم اختياره لأنه الأعلى صلة بسؤالك حسب البحث الهجين."


def sort_sources_by_intent(sources: list, intent: str) -> list:
    def priority(src):
        src_type = src.get("type")
        if intent == "statistics":
            if src_type == "statistics":
                return 0
            if src_type in ("department", "departments"):
                return 1
        elif intent == "staff":
            if src_type == "staff":
                return 0
        elif intent == "courses":
            if src_type == "courses":
                return 0
        return 2

    return sorted(sources, key=lambda s: (priority(s), -float(s.get("score", 0) or 0)))


def build_disambiguation_candidates(query: str, sources: list, max_items: int = 3) -> list:
    if not is_staff_query(query):
        return []

    staff_rows = [s for s in sources if s.get("type") == "staff"]
    if len(staff_rows) < 2:
        return []

    top_score = float(staff_rows[0].get("score", 0) or 0)
    second_score = float(staff_rows[1].get("score", 0) or 0)
    if abs(top_score - second_score) > 0.24:
        return []

    names = []
    for row in staff_rows:
        name = row.get("full_name") or row.get("title_ar") or ""
        if name and name not in names:
            names.append(name)
        if len(names) >= max_items:
            break

    return [f"اعرض بيانات {n} كاملة" for n in names]


def prepare_sources_for_view(sources: list, query: str = "") -> list:
    intent = infer_query_intent(query)
    ordered_sources = sort_sources_by_intent(sources, intent)
    rendered = []
    for src in ordered_sources:
        text = src.get("text") or src.get("text_ar") or src.get("description_en", "")
        score = float(src.get("score", 0) or 0)
        rendered.append({
            "title": src.get("title") or src.get("title_ar") or "محتوى",
            "score_text": f"{score * 100:.1f}%",
            "type": src.get("type"),
            "category": src.get("category"),
            "level": src.get("level"),
            "semester": src.get("semester"),
            "reason": source_reason(src, intent),
            "snippet_html": source_to_html(text),
        })
    return rendered


@bp.route("/", methods=["GET", "POST"])
def index():
    rag = current_app.config["RAG_ENGINE"]
    ollama_cache = current_app.config["OLLAMA_CACHE"]
    answer_cache = current_app.config["ANSWER_CACHE"]

    query = ""
    answer = ""
    answer_html = ""
    sources = []
    rendered_sources = []
    disambiguation_questions = []
    mode = "retrieval"
    error = ""

    ollama_running = ollama_cache.get("status")
    if ollama_running is None:
        ollama_running = check_ollama()
        ollama_cache.set("status", ollama_running)

    if request.method == "POST":
        query = (request.form.get("query") or "").strip()
        if query:
            cached = answer_cache.get(query)
            if cached:
                answer = cached["answer"]
                answer_html = cached["answer_html"]
                sources = cached["sources"]
                rendered_sources = cached.get("rendered_sources", [])
                disambiguation_questions = cached.get("disambiguation_questions", [])
                mode = cached["mode"]
            else:
                try:
                    sources = rag.search(query, top_k=6)
                    if sources and sources[0].get("type") == "courses":
                        answer = "\n".join([f"- {c}" for c in sources[0].get("courses", [])])
                        mode = "retrieval"
                    elif sources and sources[0].get("type") == "staff":
                        answer = compose_staff_answer(query, sources[0])
                        mode = "retrieval"
                    elif ollama_running:
                        try:
                            answer = generate_answer(query, sources)
                            mode = "ollama"
                        except Exception:
                            answer = format_retrieved_answer(sources)
                            mode = "retrieval"
                    else:
                        answer = format_retrieved_answer(sources)
                        mode = "retrieval"

                    answer_html = answer_to_html(answer)
                    rendered_sources = prepare_sources_for_view(sources, query)
                    disambiguation_questions = build_disambiguation_candidates(query, sources)
                    answer_cache.set(query, {
                        "answer": answer,
                        "answer_html": answer_html,
                        "sources": sources,
                        "rendered_sources": rendered_sources,
                        "disambiguation_questions": disambiguation_questions,
                        "mode": mode,
                    })
                except Exception as ex:
                    error = f"حدث خطأ أثناء معالجة السؤال: {ex}"

    return render_template(
        "index.html",
        query=query,
        answer=answer,
        answer_html=answer_html,
        sources=sources,
        rendered_sources=rendered_sources,
        disambiguation_questions=disambiguation_questions,
        mode=mode,
        error=error,
        ollama_running=ollama_running,
        model_name=OLLAMA_MODEL,
    )


@bp.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.svg"), code=302)


@bp.route("/clear-history", methods=["GET"])
def clear_history():
    answer_cache = current_app.config["ANSWER_CACHE"]
    answer_cache.clear()
    return redirect(url_for("main.index"), code=302)
