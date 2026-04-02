import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from src.loader import load_data
from src.summarizer import summarize_visualization

st.title("🌉 Bridge Users: Cross-Community Connections")

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_data
def get_data():
    return load_data("notebook/data.jsonl")

df = get_data()

if df.empty:
    st.warning("No data available.")
    st.stop()

required_columns = {"subreddit", "author"}
missing_columns = sorted(required_columns - set(df.columns))

if missing_columns:
    st.error(
        "Missing required columns in notebook/data.jsonl: "
        + ", ".join(missing_columns)
    )
    st.stop()

# ----------------------------------
# Clean Authors
# ----------------------------------
df = df.copy()

df = df[
    (df["author"].notna()) &
    (~df["author"].isin(["[deleted]", "[removed]", "AutoModerator"]))
]

if df.empty:
    st.warning("No valid authors found.")
    st.stop()

# ----------------------------------
# Sidebar Controls
# ----------------------------------
st.sidebar.subheader("⚙️ Network Controls")

min_shared_users = st.sidebar.slider("Min Shared Authors", 1, 10, 2)
top_n_subs = st.sidebar.slider("Top Subreddits", 5, 20, 10)

# ----------------------------------
# Author → Subreddit Mapping
# ----------------------------------
author_subs = (
    df.groupby("author")["subreddit"]
    .apply(lambda x: list(set(x)))
)

# ----------------------------------
# Build Overlap
# ----------------------------------
from collections import defaultdict
import itertools

overlap_counts = defaultdict(int)

for subs in author_subs:
    if len(subs) < 2:
        continue
    for a, b in itertools.combinations(subs, 2):
        overlap_counts[(a, b)] += 1
        overlap_counts[(b, a)] += 1

overlap_df = pd.DataFrame([
    {"subreddit_1": k[0], "subreddit_2": k[1], "shared_authors": v}
    for k, v in overlap_counts.items()
])

if overlap_df.empty:
    st.info("No cross-community author overlap found.")
    st.stop()

# ----------------------------------
# Filter Strong Connections
# ----------------------------------
overlap_df = overlap_df[overlap_df["shared_authors"] >= min_shared_users]

if overlap_df.empty:
    st.warning("No connections above threshold.")
    st.stop()

# ----------------------------------
# Select Top Subreddits
# ----------------------------------
top_subs = (
    overlap_df.groupby("subreddit_1")["shared_authors"]
    .sum()
    .sort_values(ascending=False)
    .head(top_n_subs)
    .index
)

overlap_df = overlap_df[
    overlap_df["subreddit_1"].isin(top_subs)
]

# ----------------------------------
# Build Graph
# ----------------------------------
G = nx.Graph()

for _, row in overlap_df.iterrows():
    if row["subreddit_1"] != row["subreddit_2"]:  # avoid self-loop
        G.add_edge(
            row["subreddit_1"],
            row["subreddit_2"],
            weight=row["shared_authors"]
        )

if G.number_of_nodes() == 0:
    st.warning("Graph is empty after filtering.")
    st.stop()

# ----------------------------------
# Centrality Metrics
# ----------------------------------
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

centrality_df = pd.DataFrame({
    "subreddit": list(degree_centrality.keys()),
    "degree": list(degree_centrality.values()),
    "betweenness": list(betweenness_centrality.values())
}).sort_values("betweenness", ascending=False)

# ----------------------------------
# Network Visualization (Plotly)
# ----------------------------------
st.subheader("🌐 Subreddit Connection Network")

pos = nx.spring_layout(G, seed=42)

edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x, node_y, node_size, text = [], [], [], []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    text.append(node)

    # size based on centrality
    size = 10 + betweenness_centrality.get(node, 0) * 200
    node_size.append(size)

fig = go.Figure()

# edges
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode="lines",
    line=dict(width=0.5),
    hoverinfo="none"
))

# nodes
fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=text,
    textposition="top center",
    marker=dict(size=node_size),
))

fig.update_layout(
    title="Subreddit Network (Shared Authors)",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
st.caption(
    summarize_visualization(
        "Network: Subreddit Connection (Shared Authors)",
        data=overlap_df,
        extra="Edges reflect shared authors across subreddits.",
    )
)

# ----------------------------------
# Bridge Authors
# ----------------------------------
st.subheader("🌉 Top Bridge Authors")

author_stats = (
    df.groupby("author")
    .agg(
        num_subreddits=("subreddit", "nunique"),
        total_posts=("subreddit", "count")
    )
    .reset_index()
)

bridge_authors = author_stats[author_stats["num_subreddits"] >= 2]

top_authors = bridge_authors.sort_values(
    ["num_subreddits", "total_posts"],
    ascending=False
).head(10)

st.dataframe(top_authors)
st.caption(
    summarize_visualization(
        "Table: Top Bridge Authors",
        data=top_authors,
        extra="Authors active in multiple communities.",
    )
)

# ----------------------------------
# Insights
# ----------------------------------
st.subheader("🧠 Insights")

if not centrality_df.empty:
    st.write("### 🌐 Most Connected Subreddits")
    st.dataframe(centrality_df.head(5))
    st.caption(
        summarize_visualization(
            "Table: Most Connected Subreddits",
            data=centrality_df.head(5),
            extra="Subreddits with highest centrality in the overlap graph.",
        )
    )

    st.write("### 🌉 Strong Bridge Authors")
    st.dataframe(top_authors.head(5))
    st.caption(
        summarize_visualization(
            "Table: Strong Bridge Authors",
            data=top_authors.head(5),
            extra="Top authors spanning multiple subreddits.",
        )
    )
else:
    st.info("Not enough data for insights.")
