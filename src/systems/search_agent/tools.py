"""Tool schemas exposed to the model via the Responses API."""

from typing import Any, Dict


WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the web via Brave's LLM Context API. Returns ranked, "
        "pre-extracted page content with stable numeric IDs you can cite."
    ),
    # Strict mode requires every property to be listed in `required`; for
    # optional inputs the type is widened to include "null" and the caller
    # is expected to pass null when the field is unused.
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "count", "freshness"],
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Keyword / noun-phrase search query in the style of a "
                    "search engine (e.g. 'tallest mountains in the world'), "
                    "not a natural-language question. Brave operators are "
                    "supported: quotes for exact phrases, site:domain, "
                    "-excluded_term, filetype:pdf. Max 400 chars and 50 words."
                ),
            },
            "count": {
                "type": ["integer", "null"],
                "description": "Max number of results to return (1-10). Pass null to use the server default (5).",
                "minimum": 1,
                "maximum": 10,
            },
            "freshness": {
                "type": ["string", "null"],
                "description": "Recency filter; pass null for no filter. 'pd'=past day, 'pw'=past week, 'pm'=past month, 'py'=past year.",
                "enum": ["pd", "pw", "pm", "py", None],
            },
        },
    },
    "strict": True,
}
