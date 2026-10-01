"""Micro-Jev: one-pass packed decision encoder for RAG control.

    import microjev
    d = microjev.load()                          # released weights (HF "sdmlai/micro-jev")
    d = microjev.load("runs/micro-jev-dev")      # or a local training run
    res = d.run(query=q, passages=chunks,
                decisions=["relevance", "sufficient", microjev.Q("Is the query time-sensitive?", ["yes", "no"])])
"""

from jevcore import __version__
from jevcore.decider import Decider, Q

__all__ = ["Decider", "Q", "load", "__version__"]


def load(path: str | None = None, device: str | None = None, calibration: str = "in",
         max_len: int | None = None) -> Decider:
    return Decider.from_pretrained(path, device, calibration, max_len)
