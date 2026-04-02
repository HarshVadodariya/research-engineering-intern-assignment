import pandas as pd

def _normalize_jsonl(df: pd.DataFrame) -> pd.DataFrame:
    if "data" in df.columns:
        normalized = pd.json_normalize(df["data"], sep="_")
        return normalized
    return df

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset safely.

    Args:
        path (str): File path

    Returns:
        pd.DataFrame
    """
    try:
        if path.endswith(".jsonl"):
            df = pd.read_json(path, lines=True)
            return _normalize_jsonl(df)
        if path.endswith(".json"):
            df = pd.read_json(path)
            return _normalize_jsonl(df)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
