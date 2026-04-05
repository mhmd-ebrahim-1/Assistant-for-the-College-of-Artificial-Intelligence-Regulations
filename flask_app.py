import os
import html
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify
from rag_system import LaihaRAG, check_ollama, generate_answer, OLLAMA_MODEL, is_staff_query, compose_staff_answer

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

rag = LaihaRAG("./index")
rag.ensure_index(["data.json", "data2.json"])

# Simple in-memory caches to reduce repeated latency on local usage.
OLLAMA_STATUS_TTL = 8
ANSWER_CACHE_TTL = 300
_ollama_cache = {"value": None, "ts": 0.0}
_answer_cache = {}

SUGGESTED_QUESTION_GROUPS = [
    {
        "id": "regs",
        "title": "اللوائح",
        "items": [
            "كم ساعة للتخرج؟",
            "ما شروط مرتبة الشرف؟",
            "ما درجة النجاح في المقرر؟",
            "مواد المستوى الثاني الفصل الأول",
        ],
    },
    {
        "id": "staff",
        "title": "هيئة التدريس",
        "items": [
            "مين وكيل الكلية للدراسات العليا والبحوث؟",
            "اعرض بيانات د. فاطمة محمد طلعت المرسي كاملة",
            "ما إنجازات د. فاطمة محمد طلعت المرسي؟",
            "ما تخصص حسن سعد حسن عبداللطيف محمود ابو منيسي؟",
        ],
    },
    {
        "id": "stats",
        "title": "إحصائيات",
        "items": [
            "كم عدد أعضاء هيئة التدريس في الكلية؟",
            "كم عدد المعيدين في الكلية؟",
            "كم عدد أعضاء قسم علوم البيانات؟",
            "اعرض إحصائيات الكلية حسب القسم",
        ],
    },
]


def format_retrieved_answer(chunks):
    lines = []
    for chunk in chunks[:5]:
        text = chunk.get("text") or chunk.get("text_ar") or chunk.get("description_en", "")
        text = (text or "").replace(" | ", "\n")
        if text.strip():
            lines.append(f"- {text}")
    return "\n\n".join(lines) if lines else "لا توجد نتيجة مناسبة."


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
    return "تم اختياره لأنه الأعلى صلةً بسؤالك حسب البحث الهجين."


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

    return sorted(
        sources,
        key=lambda s: (priority(s), -float(s.get("score", 0) or 0)),
    )


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


def build_dynamic_questions(max_items: int = 12) -> list:
    questions = []
    seen = set()

    chunks = getattr(rag, "chunks", None) or []
    for row in chunks:
        if len(questions) >= max_items:
            break

        t = row.get("type")
        if t == "staff":
            name = row.get("full_name") or row.get("title_ar")
            if name:
                q = f"اعرض بيانات {name} كاملة"
                if q not in seen:
                    seen.add(q)
                    questions.append(q)
        elif t == "statistics":
            q = "اعرض إحصائيات الكلية حسب القسم"
            if q not in seen:
                seen.add(q)
                questions.append(q)
        elif t == "department":
            dept = row.get("department")
            if dept:
                q = f"كم عدد أعضاء قسم {dept}؟"
                if q not in seen:
                    seen.add(q)
                    questions.append(q)
        elif t == "courses":
            level = row.get("level")
            semester = row.get("semester")
            if level and semester:
                q = f"مواد المستوى {level} الفصل {semester}"
                if q not in seen:
                    seen.add(q)
                    questions.append(q)

    return questions[:max_items]


def get_ollama_status_cached() -> bool:
    now = time.time()
    if _ollama_cache["value"] is not None and (now - _ollama_cache["ts"]) < OLLAMA_STATUS_TTL:
        return _ollama_cache["value"]
    value = check_ollama()
    _ollama_cache["value"] = value
    _ollama_cache["ts"] = now
    return value


def get_answer_cache(query: str):
    now = time.time()
    row = _answer_cache.get(query)
    if not row:
        return None
    if (now - row["ts"]) > ANSWER_CACHE_TTL:
        _answer_cache.pop(query, None)
        return None
    return row["payload"]


def set_answer_cache(query: str, payload: dict):
    _answer_cache[query] = {
        "ts": time.time(),
        "payload": payload,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    answer = ""
    answer_html = ""
    sources = []
    rendered_sources = []
    disambiguation_questions = []
    mode = "retrieval"
    error = ""

    ollama_running = get_ollama_status_cached()

    if request.method == "POST":
        query = (request.form.get("query") or "").strip()
        if query:
            cached = get_answer_cache(query)
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
                    set_answer_cache(query, {
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
        suggested_question_groups=SUGGESTED_QUESTION_GROUPS,
    )


@app.route("/api/suggest-more", methods=["GET"])
def suggest_more():
    items = build_dynamic_questions(max_items=12)
    return jsonify({"items": items})


@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.svg"), code=302)


@app.route("/clear-history", methods=["GET"])
def clear_history():
    _answer_cache.clear()
    return redirect(url_for("index"), code=302)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
