import re
from typing import Optional


def extract_tag_val(text: str, tag: str, use_rest_if_no_end: bool = False) -> Optional[str]:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no match and flag is set, try to find opening tag and use rest of text
    if use_rest_if_no_end:
        pattern_open = f"<{tag}>(.*)"
        match = re.search(pattern_open, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return None
