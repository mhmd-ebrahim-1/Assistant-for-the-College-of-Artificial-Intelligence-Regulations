from rag_ksa_ai.data.loader import flatten_values


def format_retrieved_answer(chunks):
    lines = []
    for chunk in chunks[:5]:
        text = chunk.get("text") or chunk.get("text_ar") or chunk.get("description_en", "")
        text = (text or "").replace(" | ", "\n")
        if text.strip():
            lines.append(f"- {text}")
    return "\n\n".join(lines) if lines else "لا توجد نتيجة مناسبة."


def compose_staff_answer(query: str, staff_chunk: dict) -> str:
    profile = staff_chunk.get("staff_profile") or {}
    name = profile.get("full_name") or staff_chunk.get("title_ar") or staff_chunk.get("title") or "غير محدد"
    position = profile.get("position") or "غير محدد"
    role = profile.get("current_role")
    department = profile.get("department")
    spec = profile.get("specialization_specific") or profile.get("specialization_general")
    email = profile.get("email")

    q = query.strip()
    birth_date = profile.get("birth_date")
    appointment_date = profile.get("appointment_date")
    h_index = profile.get("h_index")
    publications = profile.get("publications_count")

    achievements = flatten_values(profile.get("achievements"))
    if not achievements:
        achievements = flatten_values((profile.get("additional_info") or {}).get("achievements"))

    interests = flatten_values(profile.get("research_interests"))
    memberships = flatten_values(profile.get("memberships"))

    lines = [f"بيانات {name}:"]
    lines.append(f"- الوظيفة: {position}")
    if role:
        lines.append(f"- الدور الحالي: {role}")
    if department:
        lines.append(f"- القسم/الجهة: {department}")
    if spec:
        lines.append(f"- التخصص: {spec}")
    if email and "لم يتم" not in str(email):
        lines.append(f"- البريد الإلكتروني: {email}")
    if birth_date:
        lines.append(f"- تاريخ الميلاد: {birth_date}")
    if appointment_date:
        lines.append(f"- تاريخ التعيين: {appointment_date}")
    if h_index:
        lines.append(f"- H-Index: {h_index}")
    if publications:
        lines.append(f"- عدد الأبحاث: {publications}")

    if achievements:
        lines.append("- أبرز الإنجازات:")
        lines.extend([f"  - {a}" for a in achievements[:4]])

    if interests:
        lines.append("- الاهتمامات البحثية:")
        lines.extend([f"  - {i}" for i in interests[:6]])

    if memberships:
        lines.append("- العضويات:")
        lines.extend([f"  - {m}" for m in memberships[:4]])

    asks_email = any(k in q for k in ["إيميل", "ايميل", "email", "البريد"])
    asks_spec = any(k in q for k in ["تخصص", "مجال", "research", "اهتمام"])
    asks_role = any(k in q for k in ["وكيل", "عميد", "أمين", "امين", "منصب", "دور"])

    if asks_email and not (email and "لم يتم" not in str(email)):
        lines.insert(1, "- ملاحظة: لا يوجد بريد إلكتروني متاح في البيانات الحالية.")

    if asks_spec and not spec:
        lines.insert(1, "- ملاحظة: لا يوجد تخصص تفصيلي متاح في البيانات الحالية.")

    if asks_role and not role:
        lines.insert(1, "- ملاحظة: لا يوجد دور إداري محدد لهذا الاسم في البيانات الحالية.")

    return "\n".join(lines)
