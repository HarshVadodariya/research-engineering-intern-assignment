import pandas as pd

def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily time-series.

    Returns:
        pd.DataFrame
    """
    if df.empty:
        return df

    # TODO: implement logic
    return df