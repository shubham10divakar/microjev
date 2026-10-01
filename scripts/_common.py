"""Helpers shared by the scripts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Tables contain "→" / "Δ"; the default Windows console code page (cp1252) can't print them.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import yaml  # noqa: E402


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def device_arg(name: str | None) -> str:
    return name or ("cuda" if torch.cuda.is_available() else "cpu")


def nano_path() -> Path:
    """The sibling nano_jev checkout (code_repo/nano_jev), for baselines."""
    return ROOT.parent / "nano_jev"
