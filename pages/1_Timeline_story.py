import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.loader import load_data
from src.summarizer import summarize_visualization

st.title("📈 Timeline Story: Community Activity & Spikes")

# ----------------------------------
# Load Data (cached)
# ----------------------------------
@st.cache_data
def get_data():
    return load_data("notebook/data.jsonl")

df = get_data()

if df.empty:
    st.warning("No data available.")
    st.stop()

required_columns = {"subreddit", "created_utc", "title", "selftext"}
missing_columns = sorted(required_columns - set(df.columns))

if missing_columns:
    st.error(
        "Missing required columns in notebook/data.jsonl: "
        + ", ".join(missing_columns)
    )
    st.stop()

# ----------------------------------
# Ensure datetime
# ----------------------------------
df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
df = df.dropna(subset=["created_utc", "subreddit"])

# ----------------------------------
# Sidebar Controls
# ----------------------------------
st.sidebar.subheader("⚙️ Timeline Controls")

subreddit_options = sorted(df["subreddit"].dropna().unique())
selected_subs = st.sidebar.multiselect(
    "Subreddits",
    options=subreddit_options,
    default=subreddit_options[:5]
)

min_date = df["created_utc"].min()
max_date = df["created_utc"].max()

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date]
)

keyword = st.sidebar.text_input("Keyword")

top_n = st.sidebar.slider("Top Subreddits", 3, 10, 5)
use_rolling = st.sidebar.checkbox("Show 7-day Rolling Average", value=True)
spike_threshold = st.sidebar.slider("Spike Threshold (%)", 50, 300, 100)

# ----------------------------------
# Time-Series Construction
# ----------------------------------
filtered_df = df.copy()

if selected_subs:
    filtered_df = filtered_df[filtered_df["subreddit"].isin(selected_subs)]

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["created_utc"] >= str(date_range[0])) &
        (filtered_df["created_utc"] <= str(date_range[1]))
    ]

if keyword:
    filtered_df = filtered_df[
        filtered_df["title"].fillna("").str.contains(keyword, case=False, na=False)
        | filtered_df["selftext"].fillna("").str.contains(keyword, case=False, na=False)
    ]

if filtered_df.empty:
    st.warning("No data after applying filters.")
    st.stop()

filtered_df["date"] = filtered_df["created_utc"].dt.floor("D")

ts_df = (
    filtered_df.groupby(["subreddit", "date"])
    .size()
    .reset_index(name="post_count")
)

# Fill missing dates per subreddit
def fill_missing_dates(group):
    full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
    group = group.set_index("date").reindex(full_range, fill_value=0)
    group.index.name = "date"
    group = group.reset_index()
    group["subreddit"] = group["subreddit"].iloc[0]
    return group

ts_df = ts_df.groupby("subreddit", group_keys=False).apply(fill_missing_dates)

# ----------------------------------
# Rolling Average
# ----------------------------------
ts_df = ts_df.sort_values(["subreddit", "date"])

ts_df["rolling_avg"] = (
    ts_df.groupby("subreddit")["post_count"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

# ----------------------------------
# Spike Detection
# ----------------------------------
ts_df["prev_avg"] = ts_df.groupby("subreddit")["rolling_avg"].shift(1)
ts_df["prev_avg"] = ts_df["prev_avg"].fillna(1)

ts_df["pct_increase"] = (
    (ts_df["post_count"] - ts_df["prev_avg"]) / ts_df["prev_avg"]
) * 100

ts_df["is_spike"] = ts_df["pct_increase"] > spike_threshold

spike_df = ts_df[ts_df["is_spike"]]

# ----------------------------------
# Select Top Subreddits
# ----------------------------------
top_subs = (
    ts_df.groupby("subreddit")["post_count"]
    .sum()
    .sort_values(ascending=False)
    .head(top_n)
    .index
)

plot_df = ts_df[ts_df["subreddit"].isin(top_subs)]
spike_plot = spike_df[spike_df["subreddit"].isin(top_subs)]

# ----------------------------------
# Plot
# ----------------------------------
fig = px.line(
    plot_df,
    x="date",
    y="post_count",
    color="subreddit",
    title="Daily Posting Activity"
)

# Add rolling average
if use_rolling:
    for sub in top_subs:
        sub_df = plot_df[plot_df["subreddit"] == sub]
        fig.add_scatter(
            x=sub_df["date"],
            y=sub_df["rolling_avg"],
            mode="lines",
            name=f"{sub} (7d avg)",
            line=dict(dash="dash")
        )

# Add spike markers
fig.add_scatter(
    x=spike_plot["date"],
    y=spike_plot["post_count"],
    mode="markers",
    name="Spikes",
    marker=dict(size=8, symbol="x")
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    summarize_visualization(
        "Timeline: Daily Posting Activity",
        data=plot_df,
        extra="Shows daily post counts by subreddit after sidebar filters.",
    )
)

# ----------------------------------
# Insights
# ----------------------------------
st.subheader("🧠 Insights")

if spike_df.empty:
    st.info("No spikes detected with current threshold.")
else:
    # First spike per subreddit
    first_spikes = (
        spike_df.sort_values("date")
        .groupby("subreddit")
        .first()
        .reset_index()
    )

    earliest = first_spikes.sort_values("date").head(3)

    st.write("### 🏁 Early Spike Leaders")
    st.dataframe(earliest[["subreddit", "date", "post_count"]])
    st.caption(
        summarize_visualization(
            "Table: Early Spike Leaders",
            data=earliest[["subreddit", "date", "post_count"]],
            extra="First detected spike per subreddit.",
        )
    )

    # Strongest spikes
    strongest = spike_df.sort_values("pct_increase", ascending=False).head(5)

    st.write("### 🔥 Strongest Spikes")
    st.dataframe(
        strongest[["subreddit", "date", "post_count", "pct_increase"]]
    )
    st.caption(
        summarize_visualization(
            "Table: Strongest Spikes",
            data=strongest[["subreddit", "date", "post_count", "pct_increase"]],
            extra="Largest percent increases in posting activity.",
        )
    )
