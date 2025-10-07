from typing import NamedTuple


class PredictionResult(NamedTuple):
    """Query complexity prediction result."""
    query: str
    is_simple_prob: float
    is_simple: bool
    confidence: float
    infer_time: float
