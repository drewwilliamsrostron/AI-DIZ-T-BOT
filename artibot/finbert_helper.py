"""FinBERT sentiment with cached fallback."""

from __future__ import annotations

import os
from pathlib import Path

from artibot.auto_install import require


_CACHE = Path.home() / ".cache" / "artibot" / "finbert"
_CACHE.mkdir(parents=True, exist_ok=True)

NO_HEAVY = os.environ.get("NO_HEAVY") == "1"


def _load_pipeline():
    """Return FinBERT sentiment pipeline from Hugging Face."""

    require("huggingface-hub[hf_xet]", "huggingface_hub")
    require("transformers==4.52.4")
    from transformers import pipeline

    return pipeline(
        "text-classification", model="ProsusAI/finbert", cache_dir=str(_CACHE)
    )


try:  # pragma: no cover - optional network
    if NO_HEAVY:
        raise RuntimeError("skip heavy")
    _PIPE = _load_pipeline()

    def score(text: str) -> float:
        try:
            return float(_PIPE(text)[0]["score"])
        except Exception:
            return 0.0

except Exception:

    def score(text: str) -> float:  # type: ignore
        return 0.0
