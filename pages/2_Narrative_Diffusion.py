import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx

from src.loader import load_data
from src.summarizer import summarize_visualization

st.title("🔁 Narrative Diffusion: Cross-Community Flow")

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

required_columns = {"subreddit"}
missing_columns = sorted(required_columns - set(df.columns))

if missing_columns:
    st.error(
        "Missing required columns in notebook/data.jsonl: "
        + ", ".join(missing_columns)
    )
    st.stop()

# ----------------------------------
# Basic Cleaning
# ----------------------------------
df = df.copy()

df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
df = df.dropna(subset=["subreddit"])

# ----------------------------------
# Sidebar Controls
# ----------------------------------
st.sidebar.subheader("⚙️ Diffusion Controls")

min_edge_weight = st.sidebar.slider("Min Flow Strength", 1, 10, 2)
top_n_nodes = st.sidebar.slider("Top Subreddits", 5, 20, 10)

# ----------------------------------
# CROSSPOST ANALYSIS
# ----------------------------------
st.subheader("🔁 Crosspost Flow")

# Identify crossposts
crossposts = df.dropna(subset=["crosspost_parent"]).copy()

if crossposts.empty:
    st.info("No crosspost data available in dataset.")
else:
    # Map id -> subreddit
    if "id" in df.columns:
        id_to_sub = df.set_index("id")["subreddit"].to_dict()
    else:
        id_to_sub = {}

    def get_source(parent):
        try:
            pid = parent.split("_")[-1]
            return id_to_sub.get(pid, "unknown_source")
        except:
            return "unknown_source"

    crossposts["source_subreddit"] = crossposts["crosspost_parent"].apply(get_source)
    crossposts["destination_subreddit"] = crossposts["subreddit"]

    # Build flow table
    flow_df = (
        crossposts.groupby(["source_subreddit", "destination_subreddit"])
        .size()
        .reset_index(name="flow")
    )

    # Filter weak edges
    flow_df = flow_df[flow_df["flow"] >= min_edge_weight]

    if flow_df.empty:
        st.warning("No strong crosspost flows found.")
    else:
        # Select top nodes
        top_nodes = (
            flow_df.groupby("source_subreddit")["flow"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n_nodes)
            .index
        )

        flow_df = flow_df[flow_df["source_subreddit"].isin(top_nodes)]

        # ----------------------------------
        # Heatmap
        # ----------------------------------
        st.markdown("### 🔥 Crosspost Heatmap")

        heatmap = flow_df.pivot_table(
            index="source_subreddit",
            columns="destination_subreddit",
            values="flow",
            fill_value=0
        )

        fig = px.imshow(
            heatmap,
            title="Cross-Community Flow (Source → Destination)",
            aspect="auto"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            summarize_visualization(
                "Heatmap: Crosspost Flow",
                data=heatmap,
                extra="Crosspost flow counts between source and destination subreddits.",
            )
        )

        # ----------------------------------
        # Network Graph
        # ----------------------------------
        st.markdown("### 🌐 Network Graph")

        G = nx.DiGraph()

        for _, row in flow_df.iterrows():
            G.add_edge(
                row["source_subreddit"],
                row["destination_subreddit"],
                weight=row["flow"]
            )

        if G.number_of_nodes() == 0:
            st.warning("Graph is empty.")
        else:
            pos = nx.spring_layout(G, seed=42)

            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x, node_y, text = [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                text.append(node)

            import plotly.graph_objects as go

            fig_net = go.Figure()

            fig_net.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=0.5),
                hoverinfo="none"
            ))

            fig_net.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=text,
                textposition="top center",
                marker=dict(size=12),
            ))

            fig_net.update_layout(
                title="Subreddit Diffusion Network",
                showlegend=False
            )

            st.plotly_chart(fig_net, use_container_width=True)
            st.caption(
                summarize_visualization(
                    "Network: Subreddit Diffusion",
                    data=flow_df,
                    extra="Directed crosspost flows among top subreddits.",
                )
            )

# ----------------------------------
# URL SHARING ANALYSIS
# ----------------------------------
st.subheader("🔗 Shared URL Diffusion")

from urllib.parse import urlparse

def extract_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except:
        return np.nan

if "url" not in df.columns:
    st.info("No URL column found.")
else:
    df["domain"] = df["url"].apply(extract_domain)
    df_urls = df.dropna(subset=["domain"])

    if df_urls.empty:
        st.info("No valid URLs found.")
    else:
        url_group = (
            df_urls.groupby("url")
            .agg({
                "subreddit": lambda x: list(set(x)),
                "domain": "first",
                "url": "count"
            })
            .rename(columns={"url": "frequency"})
            .reset_index()
        )

        url_group["num_subreddits"] = url_group["subreddit"].apply(len)

        shared_urls = url_group[url_group["num_subreddits"] >= 2]

        if shared_urls.empty:
            st.info("No cross-community URLs found.")
        else:
            st.markdown("### 🔥 Top Shared URLs")

            st.dataframe(
                shared_urls.sort_values("frequency", ascending=False)
                .head(10)[["domain", "frequency", "num_subreddits"]]
            )
            st.caption(
                summarize_visualization(
                    "Table: Top Shared URLs",
                    data=shared_urls.sort_values("frequency", ascending=False)
                    .head(10)[["domain", "frequency", "num_subreddits"]],
                    extra="Most shared URLs across multiple communities.",
                )
            )

# ----------------------------------
# Insights
# ----------------------------------
st.subheader("🧠 Insights")

if 'flow_df' in locals() and not flow_df.empty:
    top_sources = (
        flow_df.groupby("source_subreddit")["flow"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )

    top_dest = (
        flow_df.groupby("destination_subreddit")["flow"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )

    st.write("### 📡 Top Source Communities")
    st.write(top_sources)
    st.caption(
        summarize_visualization(
            "Table: Top Source Communities",
            data=top_sources,
            extra="Communities that originate the most crosspost flow.",
        )
    )

    st.write("### 📢 Top Amplifier Communities")
    st.write(top_dest)
    st.caption(
        summarize_visualization(
            "Table: Top Amplifier Communities",
            data=top_dest,
            extra="Communities that receive the most crosspost flow.",
        )
    )
else:
    st.info("Not enough data for insights.")
