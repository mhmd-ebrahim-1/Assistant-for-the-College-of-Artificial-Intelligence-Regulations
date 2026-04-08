import json
from pathlib import Path


def load_json_data(path: str = "data.json") -> list:
    with open(path, encoding="utf-8-sig") as f:
        raw = json.load(f)
    source_name = Path(path).name
    return normalize_data_records(raw, source_name=source_name)


def flatten_values(value):
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_values(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_values(item))
        return out
    return [str(value)]


def _build_staff_entry(staff: dict) -> dict:
    normalized = dict(staff or {})

    if not normalized.get("specialization_specific") and normalized.get("specialization"):
        normalized["specialization_specific"] = normalized.get("specialization")

    additional = normalized.get("additional_info")
    if isinstance(additional, dict):
        for k, v in additional.items():
            if k not in normalized:
                normalized[k] = v

    if not normalized.get("current_role") and normalized.get("role"):
        normalized["current_role"] = normalized.get("role")

    if not normalized.get("full_name") and normalized.get("name"):
        normalized["full_name"] = normalized.get("name")

    text_parts = []
    for key in [
        "full_name", "full_name_en", "position", "current_role", "department",
        "specialization_general", "specialization_specific", "status", "email", "notes"
    ]:
        if normalized.get(key):
            text_parts.append(str(normalized.get(key)))

    if normalized.get("birth_date"):
        text_parts.append(f"تاريخ الميلاد: {normalized.get('birth_date')}")
    if normalized.get("appointment_date"):
        text_parts.append(f"تاريخ التعيين: {normalized.get('appointment_date')}")
    if normalized.get("h_index"):
        text_parts.append(f"H-Index: {normalized.get('h_index')}")
    if normalized.get("publications_count"):
        text_parts.append(f"عدد الأبحاث: {normalized.get('publications_count')}")

    for nested_key in [
        "education", "achievements", "research_interests", "memberships",
        "previous_positions", "certifications"
    ]:
        text_parts.extend(flatten_values(normalized.get(nested_key)))

    title = normalized.get("full_name") or normalized.get("position") or "عضو هيئة تدريس"
    subtitle = normalized.get("position") or ""
    if subtitle:
        title = f"{title} - {subtitle}"

    return {
        "type": "staff",
        "category": "faculty",
        "title": title,
        "title_ar": normalized.get("full_name") or title,
        "full_name": normalized.get("full_name"),
        "position": normalized.get("position"),
        "department": normalized.get("department"),
        "keywords": [
            "دكتور", "دكتورة", "هيئة التدريس", "معيد", "قسم", "تخصص", "ايميل",
            normalized.get("full_name", ""),
            normalized.get("full_name_en", ""),
            normalized.get("department", ""),
            normalized.get("position", ""),
            normalized.get("current_role", ""),
        ],
        "text_ar": " | ".join([p for p in text_parts if p]),
        "staff_profile": normalized,
        "source": "data2.json",
    }


def normalize_data_records(raw, source_name: str = "data.json") -> list:
    if isinstance(raw, list):
        return raw

    if not isinstance(raw, dict):
        return []

    records = []

    university = raw.get("university")
    faculty = raw.get("faculty")
    if university or faculty:
        records.append({
            "type": "overview",
            "category": "faculty_info",
            "title": f"{faculty or 'الكلية'} - معلومات عامة",
            "title_ar": f"{faculty or 'الكلية'} - معلومات عامة",
            "keywords": ["الكلية", "الجامعة", "نبذة", "معلومات", university or "", faculty or ""],
            "text_ar": " | ".join(flatten_values(raw.get("university_profile"))),
        })

    faculty_details = raw.get("faculty_details") or {}
    if isinstance(faculty_details, dict):
        records.append({
            "type": "overview",
            "category": "faculty_info",
            "title": "نبذة عن الكلية",
            "title_ar": "نبذة عن الكلية",
            "keywords": ["نبذة", "كلية الذكاء الاصطناعي", "الأقسام", "البرامج"],
            "text_ar": " | ".join(flatten_values(faculty_details)),
            "source": source_name,
        })

    university_presidents = raw.get("university_presidents") or {}
    if isinstance(university_presidents, dict):
        current_president = university_presidents.get("current_president")
        if isinstance(current_president, dict):
            records.append(_build_staff_entry({
                "full_name": current_president.get("name"),
                "position": current_president.get("title") or "رئيس الجامعة",
                "current_role": current_president.get("title") or "رئيس الجامعة",
                "status": "حالي",
                "department": university,
                "email": current_president.get("official_email") or current_president.get("email"),
                "appointment_date": current_president.get("appointment_date"),
                "academic_rank": current_president.get("academic_rank"),
                "address": current_president.get("address"),
            }))

        previous_presidents = university_presidents.get("previous_presidents") or []
        if isinstance(previous_presidents, list):
            for former in previous_presidents:
                if isinstance(former, dict):
                    records.append(_build_staff_entry({
                        "full_name": former.get("name"),
                        "position": "رئيس جامعة سابق",
                        "current_role": "رئيس جامعة سابق",
                        "status": "سابق",
                        "department": university,
                        "notes": " | ".join(flatten_values(former)),
                    }))

    president_full_profile = raw.get("president_full_profile")
    if isinstance(president_full_profile, dict):
        records.append(_build_staff_entry({
            **president_full_profile,
            "full_name": president_full_profile.get("full_name") or president_full_profile.get("name"),
            "position": president_full_profile.get("title") or "رئيس الجامعة",
            "current_role": president_full_profile.get("title") or "رئيس الجامعة",
            "status": "حالي",
            "department": president_full_profile.get("faculty") or university,
            "specialization_specific": president_full_profile.get("department"),
            "email": president_full_profile.get("official_email") or president_full_profile.get("email"),
        }))

    staff_members = raw.get("staff_members") or []
    if isinstance(staff_members, list):
        for member in staff_members:
            if isinstance(member, dict):
                records.append(_build_staff_entry(member))

    leadership = faculty_details.get("leadership") or {}
    if isinstance(leadership, dict):
        dean = leadership.get("dean")
        if isinstance(dean, dict):
            records.append(_build_staff_entry({
                "full_name": dean.get("name"),
                "position": dean.get("title") or "عميد",
                "current_role": dean.get("title"),
                "department": "كلية الذكاء الاصطناعي",
            }))

        vice_deans = leadership.get("vice_deans") or []
        if isinstance(vice_deans, list):
            for vice in vice_deans:
                if isinstance(vice, dict):
                    records.append(_build_staff_entry({
                        "full_name": vice.get("name"),
                        "position": "وكيل كلية",
                        "current_role": vice.get("role"),
                        "department": "كلية الذكاء الاصطناعي",
                    }))

        secretary = leadership.get("secretary")
        if isinstance(secretary, dict):
            records.append(_build_staff_entry({
                "full_name": secretary.get("name"),
                "position": secretary.get("role") or "أمين الكلية",
                "current_role": secretary.get("role"),
                "email": secretary.get("email"),
                "department": "كلية الذكاء الاصطناعي",
            }))

    dean_full_profile = raw.get("dean_full_profile")
    if isinstance(dean_full_profile, dict):
        records.append(_build_staff_entry({
            **dean_full_profile,
            "full_name": dean_full_profile.get("full_name") or dean_full_profile.get("name"),
            "position": dean_full_profile.get("academic_rank") or "عميد كلية",
            "current_role": dean_full_profile.get("current_position"),
            "specialization_general": " | ".join(flatten_values(dean_full_profile.get("research_interests"))),
        }))

    departments = raw.get("departments")
    if isinstance(departments, list) and departments:
        if departments and isinstance(departments[0], dict):
            dept_names = [d.get("name") for d in departments if isinstance(d, dict) and d.get("name")]
            records.append({
                "type": "departments",
                "category": "faculty_info",
                "title": "أقسام الكلية",
                "title_ar": "أقسام الكلية",
                "keywords": ["أقسام", "قسم", "الكلية"],
                "text_ar": "\n".join([f"- {d}" for d in dept_names]),
                "source": source_name,
            })

            for dept in departments:
                if not isinstance(dept, dict):
                    continue
                dept_name = dept.get("name") or "قسم"
                member_count = dept.get("member_count")
                members = dept.get("members") or []

                records.append({
                    "type": "department",
                    "category": "faculty_info",
                    "title": f"قسم {dept_name}",
                    "title_ar": f"قسم {dept_name}",
                    "department": dept_name,
                    "keywords": ["قسم", "أعضاء", dept_name],
                    "text_ar": " | ".join([
                        f"اسم القسم: {dept_name}",
                        f"عدد الأعضاء: {member_count}" if member_count is not None else "",
                        f"أسماء الأعضاء: {', '.join([m.get('full_name', '') for m in members if isinstance(m, dict) and m.get('full_name')])}",
                    ]),
                    "source": source_name,
                })

                if isinstance(members, list):
                    for member in members:
                        if isinstance(member, dict):
                            records.append(_build_staff_entry({
                                **member,
                                "department": dept_name,
                                "specialization_specific": member.get("specialization"),
                            }))
        else:
            records.append({
                "type": "departments",
                "category": "faculty_info",
                "title": "أقسام الكلية",
                "title_ar": "أقسام الكلية",
                "keywords": ["أقسام", "قسم", "الكلية"],
                "text_ar": "\n".join([f"- {d}" for d in departments]),
                "source": source_name,
            })

    administrative_staff = raw.get("administrative_staff") or []
    if isinstance(administrative_staff, list):
        for admin in administrative_staff:
            if isinstance(admin, dict):
                records.append(_build_staff_entry({
                    **admin,
                    "current_role": admin.get("position"),
                    "department": "الإدارة",
                }))

    statistics = raw.get("statistics")
    if isinstance(statistics, dict):
        records.append({
            "type": "statistics",
            "category": "faculty_info",
            "title": "إحصائيات أعضاء هيئة التدريس",
            "title_ar": "إحصائيات أعضاء هيئة التدريس",
            "keywords": ["إحصائيات", "عدد", "هيئة التدريس", "أساتذة"],
            "text_ar": " | ".join(flatten_values(statistics)),
            "source": source_name,
        })

    faculty_stats = faculty_details.get("statistics") or {}
    if isinstance(faculty_stats, dict):
        records.append({
            "type": "statistics",
            "category": "faculty_info",
            "title": "إحصائيات الكلية",
            "title_ar": "إحصائيات الكلية",
            "keywords": ["إحصائيات", "عدد", "أعضاء", "هيئة التدريس", "الكلية"],
            "text_ar": " | ".join(flatten_values(faculty_stats)),
            "source": source_name,
        })

    statistics_summary = raw.get("statistics_summary")
    if isinstance(statistics_summary, dict):
        records.append({
            "type": "statistics",
            "category": "faculty_info",
            "title": "ملخص الإحصائيات",
            "title_ar": "ملخص الإحصائيات",
            "keywords": ["إحصائيات", "ملخص", "عدد", "معيد", "مدرس", "أستاذ"],
            "text_ar": " | ".join(flatten_values(statistics_summary)),
            "source": source_name,
        })

    return records
