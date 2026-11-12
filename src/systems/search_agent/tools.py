"""Tool schemas exposed to the model via the Responses API."""

from typing import Any, Dict


WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the web via Keenable. Returns ranked, "
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
                    "Describe the information or page you need in plain language. "
                    "Supported operators: quotes for exact phrases, site:domain, "
                    "and -excluded_term. Max 400 chars and 50 words."
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
                "description": "Recency filter; pass null for no filter. 'pd'=past day, 'pw'=past week, 'pm'=past 30 days, 'py'=past 365 days (UTC date granularity).",
                "enum": ["pd", "pw", "pm", "py", None],
            },
        },
    },
    "strict": True,
}
