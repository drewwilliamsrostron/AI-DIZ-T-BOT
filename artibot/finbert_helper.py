"""Optional FinBERT sentiment helper."""

from artibot.auto_install import ensure_pkg

try:
    from finbert.sentiment import SentimentAnalyzer

    model = SentimentAnalyzer()

    def score(text: str) -> str:
        return model.predict(text)[0][0]["label"]
except ModuleNotFoundError:
    ensure_pkg("transformers")
    from transformers import pipeline

    sa = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        truncation=True,
    )

    def score(text: str) -> str:
        return sa(text, max_length=128, truncation=True)[0]["label"]
