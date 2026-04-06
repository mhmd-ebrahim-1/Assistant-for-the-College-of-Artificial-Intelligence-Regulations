import json
import urllib.request

from rag_ksa_ai.config import (
    GEN_MAX_CHARS_PER_CHUNK,
    GEN_MAX_CONTEXT_CHUNKS,
    GEN_NUM_PREDICT,
    GEN_TIMEOUT_SECONDS,
    OLLAMA_MODEL,
    OLLAMA_URL,
)


def generate_answer(query: str, retrieved_chunks: list, api_key=None) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks[:GEN_MAX_CONTEXT_CHUNKS]):
        page = chunk.get("page", "-")
        if chunk.get("type") == "courses" and chunk.get("courses"):
            text = "\n".join([f"- {c}" for c in chunk.get("courses", [])])
        else:
            text = chunk.get("text") or chunk.get("text_ar") or chunk.get("description_en", "")
        context_parts.append(f"[مقطع {i+1} - {page}]:\n{text[:GEN_MAX_CHARS_PER_CHUNK]}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""<|im_start|>system
أنت مساعد أكاديمي متخصص في لائحة كلية الذكاء الاصطناعي بجامعة كفر الشيخ.
قواعد صارمة:
1. أجب باللغة العربية فقط - ممنوع الإنجليزية
2. استخدم فقط المعلومات الموجودة في السياق
3. إذا كانت الإجابة رقماً أو شرطاً محدداً، اذكره مباشرة
4. اذكر رقم المادة إن وجد
5. لا تختصر إذا كان السؤال يطلب بيانات شخص أو تفاصيل متعددة، وقدّم النقاط في شكل قائمة واضحة
<|im_end|>
<|im_start|>user
السياق:
{context}

السؤال: {query}
<|im_end|>
<|im_start|>assistant
"""

    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.05, "num_predict": GEN_NUM_PREDICT},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())["response"].strip()


def check_ollama() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True
    except Exception:
        return False
