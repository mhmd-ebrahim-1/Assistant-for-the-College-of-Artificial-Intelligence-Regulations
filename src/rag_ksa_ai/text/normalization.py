import re
import unicodedata


def normalize_arabic(text: str) -> str:
    result = [unicodedata.normalize("NFKC", c) for c in text]
    text = "".join(result)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text
