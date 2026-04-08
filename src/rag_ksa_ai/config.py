OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b-instruct"

GEN_MAX_CONTEXT_CHUNKS = 2
GEN_MAX_CHARS_PER_CHUNK = 260
GEN_NUM_PREDICT = 160
GEN_TIMEOUT_SECONDS = 45

INDEX_DIR = "./index"
DATA_FILES = [
    "data/DrData.json",
    "data/UniData.json",
    "data.json",
    "data2.json",
]

OLLAMA_STATUS_TTL = 8
ANSWER_CACHE_TTL = 300

STAFF_QUERY_TERMS = [
    "دكتور", "دكتورة", "هيئة التدريس", "معيد", "مدرس", "استاذ", "أستاذ",
    "إيميل", "ايميل", "تخصص", "قسم", "وكيل", "عميد", "أمين", "امين",
    "رئيس", "الرئيس", "رئاسة",
    "بروفايل", "السيرة", "السيره", "بيانات", "معلومات",
]
