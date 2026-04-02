import streamlit as st
from datetime import datetime
from src.loader import load_data
import pandas as pd

st.set_page_config(
    page_title="Reddit Narrative Diffusion Dashboard",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("Reddit Political Narrative Diffusion")
st.markdown("""
Investigate how political narratives spread across Reddit communities.

Track:
- Timeline spikes
- Cross-community diffusion
- Bridge users
- Topic clusters
- Semantic similarity
""")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔎 Global Filters")

@st.cache_data

def get_data():
    return load_data("notebook/data.jsonl")

df = get_data()

if df.empty:
    st.warning("No data available.")
    st.stop()

required_columns = {"subreddit", "created_utc", "title"}
missing_columns = sorted(required_columns - set(df.columns))

if missing_columns:
    st.error(
        "Missing required columns in notebook/data.jsonl: "
        + ", ".join(missing_columns)
    )
    st.stop()

# Subreddit filter
subreddits = sorted(df["subreddit"].dropna().unique())
selected_subs = st.sidebar.multiselect(
    "Subreddits",
    options=subreddits,
    default=subreddits[:5]
)

# Date filter
df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
min_date = df["created_utc"].min()
max_date = df["created_utc"].max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("No valid created_utc values found for date filtering.")
    st.stop()

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date]
)

# Keyword filter
keyword = st.sidebar.text_input("Keyword")

# -----------------------------
# Apply Filters
# -----------------------------
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
        filtered_df["title"].str.contains(keyword, case=False, na=False)
    ]

st.success(f"Filtered rows: {len(filtered_df)}")

st.markdown("👉 Use the sidebar to explore different narrative patterns.")
