import re
from typing import Optional


def extract_tag_val(text: str, tag: str) -> Optional[str]:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
