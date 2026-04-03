# Reddit Narrative Polarization & Cross-Community Amplification Dashboard

An investigative **research-engineering dashboard** for tracing how political narratives spread, amplify, and mutate across ideologically distinct Reddit communities.

This project analyzes cross-community diffusion across subreddits such as **Liberal, Conservative, Republican, democrats, neoliberal, PoliticalDiscussion, and Anarchism** to understand:

* **when narratives emerge**
* **which communities amplify them**
* **which users act as bridges**
* **how topics evolve semantically**
* **which narratives remain siloed vs cross ideological boundaries**

The dashboard is designed as a **computational social science + investigative reporting system**, aligned with social media narrative tracing and influence analysis workflows.

---

# Live Demo

**Video Walkthrough:**  :  https://drive.google.com/drive/folders/1kZxfpY84xyl6maUaac_Fi95-Yr2ldfPF?usp=share_link

---

# Research Hypothesis

> Moderate and discussion-centric communities act as **narrative bridges**, while partisan communities act as **amplification chambers**.

This hypothesis was validated through:

* temporal spike detection
* crosspost and shared URL diffusion
* shared-author overlap
* semantic topic clustering
* topic migration lag analysis

---

# Core Story

The system investigates:

> **How political narratives emerge in one subreddit and spread into ideologically distinct communities over time.**

Key story questions:

1. Which subreddit discusses a narrative first?
2. Which communities amplify it later?
3. Do narratives stay inside echo chambers?
4. Which users bridge ideological communities?
5. Are the same events framed differently across communities?

---

# Dashboard Architecture

```text
app.py
pages/
├── 1_Timeline_Story.py
├── 2_Narrative_Diffusion.py
├── 3_Bridge_Users.py
├── 4_Topic_Map.py
└── 5_Semantic_Search.py

src/
├── loader.py
├── timeline.py
├── diffusion.py
├── network.py
├── clustering.py
├── search.py
└── summarizer.py
```

---

#  Features

## 1) Timeline Story

Visualizes:

* daily post volume per subreddit
* rolling averages
* automatic spike detection
* AI-generated plain-language summaries

### Insight

Helps identify:

> which communities discuss major narratives first.

---

## 2) Narrative Diffusion

Tracks:

* crosspost source → destination flow
* shared URLs across communities
* strongest narrative migration paths
* cross-ideology diffusion heatmaps

### Insight

Reveals:

> which communities seed vs amplify narratives.

---

## 3) Bridge Users

Network analysis of:

* shared authors
* overlap between subreddits
* bridge accounts
* centrality metrics

### Metrics

* Degree centrality
* Betweenness centrality
* PageRank

### Insight

Detects:

> users who transfer narratives across ideological boundaries.

---

### Insight

Shows:

> same event, different narrative framing.

---

## 4) Semantic Search

Embedding-based retrieval for investigative exploration.

Supports:

* zero keyword overlap retrieval
* short query handling
* related query suggestions

# ML / AI Components

## Semantic Search

* **Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Similarity:** cosine similarity
* **Library:** sentence-transformers

## Topic Clustering

* **Embeddings:** `all-MiniLM-L6-v2`
* **Reduction:** UMAP (`n_neighbors=15`, `min_dist=0.1`)
* **Clustering:** KMeans (`k=8`, user tunable)
* **Library:** sentence-transformers, umap-learn, scikit-learn

## Topic Keywords

* **Method:** TF-IDF top terms per cluster
* **Library:** scikit-learn

## Network Analysis

* **Metrics:** PageRank, betweenness centrality
* **Library:** NetworkX

## Chart Summaries

* **Method:** dynamic LLM summarization over filtered dataframe statistics
* **Use:** non-technical trend explanations

---

# Robustness & Stress Testing

The dashboard is designed to safely handle:

* empty result sets
* single subreddit filters
* disconnected graph components
* isolated users
* malformed URLs
* deleted authors
* very short search queries
* non-English text queries
* extreme cluster counts

---

# 🛠️ Setup

```bash
git clone <repo-url>
cd SimPPL
pip install -r requirements.txt
streamlit run app.py
```

# 👨‍💻 AI Usage Disclosure

AI was used for:

* scaffolding
* boilerplate acceleration
* refactoring support
* deployment troubleshooting

All analytical decisions, story selection, hypothesis testing, debugging, ML parameter tuning, and validation were manually reviewed and iteratively refined.
