from rag_ksa_ai.retrieval.scoring import is_staff_query, staff_name_match_score
from rag_ksa_ai.text.processing import prepare_text


def rerank_staff_results(results: list, query: str) -> list:
    asks_email = any(k in query for k in ["إيميل", "ايميل", "email", "البريد"])
    asks_spec = any(k in query for k in ["تخصص", "مجال", "research", "اهتمام"])

    boosted = []
    for row in results:
        score = float(row.get("score", 0.0))
        if row.get("type") == "staff":
            score += 0.2
            score += 0.8 * staff_name_match_score(query, row)
            profile = row.get("staff_profile") or {}
            if asks_email and profile.get("email") and "لم يتم" not in str(profile.get("email")):
                score += 0.15
            if asks_spec and profile.get("specialization_specific"):
                score += 0.1

        copy_row = row.copy()
        copy_row["score"] = round(score, 4)
        boosted.append(copy_row)

    boosted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return boosted


def extract_level_semester(query: str):
    level_map = {
        "المستوى الأول": 1,
        "المستوى الاول": 1,
        "المستوى الثاني": 2,
        "المستوى التاني": 2,
        "المستوى الثالث": 3,
        "المستوى الرابع": 4,
    }
    semester_map = {
        "الفصل الأول": 1,
        "الفصل الاول": 1,
        "الفصل الثاني": 2,
        "الفصل التاني": 2,
    }

    level = None
    semester = None

    for phrase, value in level_map.items():
        if phrase in query:
            level = value
            break

    for phrase, value in semester_map.items():
        if phrase in query:
            semester = value
            break

    return level, semester


def smart_filter(results: list, query: str) -> list:
    q = query.strip()
    has_courses_intent = any(k in q for k in ["مواد", "مقررات", "الخطة", "المستوى", "الفصل"])

    asks_counts = any(k in q for k in ["كم", "عدد", "إحصائيات", "احصائيات", "إجمالي", "اجمالي"])
    stats_terms = any(k in q for k in ["هيئة التدريس", "المعيد", "المعيدين", "المدرس", "الكلية", "القسم", "الأقسام", "الاقسام"])

    if asks_counts and stats_terms:
        stats = [r for r in results if r.get("type") in ("statistics", "department", "departments")]
        return stats or results

    if is_staff_query(q):
        staff = [r for r in results if r.get("type") == "staff" or r.get("category") == "faculty"]
        return staff or results

    if any(k in q for k in ["وكيل", "عميد", "أمين", "امين"]):
        staff = [r for r in results if r.get("type") == "staff"]
        leadership = [
            r for r in staff
            if any(term in prepare_text(r) for term in ["وكيل", "عميد", "أمين", "امين"])
        ]
        return leadership or staff or results

    if has_courses_intent:
        level, semester = extract_level_semester(q)
        courses = [r for r in results if r.get("type") == "courses"]
        if level is not None:
            courses = [r for r in courses if r.get("level") == level]
        if semester is not None:
            courses = [r for r in courses if r.get("semester") == semester]
        return courses or results

    if any(k in q for k in ["مرتبة الشرف", "شرف"]):
        honor = [r for r in results if "مرتبة الشرف" in (r.get("title") or "")]
        return honor or results

    if any(k in q for k in ["التخرج", "ساعة", "ساعات", "144"]):
        grad = [r for r in results if r.get("category") == "graduation"]
        return grad or results

    if any(k in q for k in ["النجاح", "راسب", "امتحان", "درجة"]):
        exams = [r for r in results if r.get("category") in ("exams", "grading")]
        return exams or results

    if any(k in q for k in ["يفصل", "فصل", "إنذار", "انذار"]):
        dismissal = [r for r in results if r.get("category") == "dismissal"]
        return dismissal or results

    return results
