from typing import Any, Literal
from langgraph.config import get_stream_writer

from systems.rag_interface import RunStreamingResponse

"""
Edit log:

14 Oct: now we are using Any for value, and just passing content and reasoning
strings, but what if we need to pass contexts, citations? –– change to pass
RunStreamingResponse instead.
"""

MSG_TYPE = Literal['general', 'custom_final_answer',
                   'custom_intermediate_step']


def pub_msg(type: MSG_TYPE, value: RunStreamingResponse):
    writer = get_stream_writer()
    writer((type, value))


def to_any(x: Any) -> Any:
    """
    DO NOT USE THIS FUNCTION UNLESS ABSOLUTELY NECESSARY.
    """
    return x
