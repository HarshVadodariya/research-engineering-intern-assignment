import networkx as nx
import pandas as pd

def build_author_network(df: pd.DataFrame) -> nx.Graph:
    """
    Build subreddit network via shared authors.
    """
    return nx.Graph()