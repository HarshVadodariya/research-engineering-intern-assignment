import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

def semantic_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Perform semantic similarity search.

    Args:
        query_embedding: embedding of query
        doc_embeddings: embeddings of documents
        df: dataframe
        top_k: number of results

    Returns:
        pd.DataFrame
    """
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    indices = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[indices].copy()
    results["similarity"] = scores[indices]

    return results