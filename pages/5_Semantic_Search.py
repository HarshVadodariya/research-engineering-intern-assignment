import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.loader import load_data

st.title("🔍 Semantic Search: Find Narratives")

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

# ----------------------------------
# Prepare Text
# ----------------------------------
required_columns = {"title", "selftext", "subreddit"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "Missing required columns in notebook/data.jsonl: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

df["full_text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")

df = df[df["full_text"].str.len() > 20]

if df.empty:
    st.warning("No usable text found.")
    st.stop()

# ----------------------------------
# Load Model (cached)
# ----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------------
# Load or Generate Embeddings
# ----------------------------------
@st.cache_data
def get_embeddings(texts):
    return model.encode(texts, batch_size=32, show_progress_bar=False)

# ⚠️ Limit size for safety
MAX_DOCS = 2000

df_search = df.copy()

if len(df_search) > MAX_DOCS:
    df_search = df_search.sample(MAX_DOCS, random_state=42).reset_index(drop=True)

embeddings = get_embeddings(df_search["full_text"].tolist())

# ----------------------------------
# Search Input
# ----------------------------------
query = st.text_input("Enter a narrative or topic:")

top_k = st.slider("Top Results", 3, 20, 5)

# ----------------------------------
# Perform Search
# ----------------------------------
if query:

    with st.spinner("Searching..."):
        query_embedding = model.encode([query])

        scores = cosine_similarity(query_embedding, embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = df_search.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]

    st.subheader("🔝 Results")

    for _, row in results.iterrows():
        st.markdown(f"""
        **Subreddit:** {row['subreddit']}  
        **Score:** {row['similarity']:.3f}  
        **Title:** {row['title']}
        """)

        if row["selftext"]:
            st.caption(row["selftext"][:200])

        st.markdown("---")

# ----------------------------------
# Similar Narrative Clusters
# ----------------------------------
if query and not results.empty:

    st.subheader("🧠 Related Communities")

    subreddit_counts = results["subreddit"].value_counts()

    st.bar_chart(subreddit_counts)

# ----------------------------------
# Insights
# ----------------------------------
st.subheader("🧠 How This Works")

st.markdown("""
- Converts text into embeddings using a transformer model
- Finds semantically similar posts (not just keyword match)
- Helps identify narratives across different wording styles

Use cases:
- Track how a specific idea spreads
- Compare framing across communities
- Discover hidden related discussions
""")