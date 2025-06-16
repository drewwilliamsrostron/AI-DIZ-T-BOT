"""Thin wrapper around FinBERT with an optional fallback pipeline."""

from artibot.auto_install import ensure_pkg

try:
    from finbert.sentiment import SentimentAnalyzer

    _MODEL = SentimentAnalyzer()

    def score(text: str) -> str:
        return _MODEL.predict(text)[0][0]["label"]

except ModuleNotFoundError:
    ensure_pkg("transformers")
    try:  # pragma: no cover - optional
        from transformers import pipeline

        _PIPE = pipeline(
            "text-classification", model="ProsusAI/finbert", truncation=True
        )

        def score(text: str) -> str:
            return _PIPE(text, max_length=128, truncation=True)[0]["label"]

    except Exception:

        def score(text: str) -> str:
            return "neutral"
