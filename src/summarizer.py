from functools import lru_cache
from typing import Any, Optional


@lru_cache(maxsize=1)
def _get_summarizer():
    from transformers import pipeline

    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarize_text(text: str) -> str:
    """
    Generate summary using a local HF model with a safe fallback.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    try:
        summarizer = _get_summarizer()
        result = summarizer(
            cleaned,
            max_length=120,
            min_length=30,
            do_sample=False,
        )
        return result[0]["summary_text"]
    except Exception:
        return cleaned[:200]


def summarize_visualization(
    title: str,
    data: Optional[Any] = None,
    extra: Optional[str] = None,
    max_rows: int = 5,
) -> str:
    """
    Build a short context string from a visualization and summarize it.
    """
    parts = [f"Visualization: {title}"]
    if extra:
        parts.append(f"Context: {extra}")

    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            parts.append(f"Rows: {len(data)}, Columns: {list(data.columns)[:10]}")
            sample = data.head(max_rows).to_csv(index=False)
            parts.append(f"Sample:\n{sample}")
        elif isinstance(data, pd.Series):
            parts.append(f"Series name: {data.name}, Length: {len(data)}")
            sample = data.head(max_rows).to_string()
            parts.append(f"Sample:\n{sample}")
    except Exception:
        pass

    context = "\n".join(parts)
    return summarize_text(context)
