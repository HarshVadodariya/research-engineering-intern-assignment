# AI Prompt Log — Research Engineering Intern Assignment

This file documents the AI-assisted components used during development.

AI was primarily used for:
- project scaffolding
- boilerplate acceleration
- refactoring support
- visualization structure suggestions
- deployment troubleshooting

All analytical decisions, topic framing, research hypothesis design, validation, debugging, and ML parameter choices were manually reviewed and iteratively refined.

---

## Prompt 1 — Project Scaffold
**Component:** Streamlit architecture

**Prompt Used:**  
Generate a production-ready Streamlit multipage project scaffold for a Reddit narrative diffusion dashboard with modular src files.

**Issue in First Output:**  
The initial version mixed page logic and analysis utilities in a single file, which would make notebook-to-production migration difficult.

**Manual Validation / Fix:**  
I split the structure into `pages/` and `src/` modules and aligned each page to a validated notebook section.

**Why Final Choice Was Better:**  
This separation improved maintainability and made it easier to migrate validated research logic into reusable production modules.

---

## Prompt 2 — Data Loader
**Component:** preprocessing

**Prompt Used:**  
Write robust Pandas code to load a Reddit JSONL dataset, normalize fields, parse timestamps, and safely handle malformed rows.

**Issue in First Output:**  
Malformed timestamps were not safely coerced, and nested missing keys created inconsistent columns.

**Manual Validation / Fix:**  
Added explicit timestamp coercion, null-safe text defaults, duplicate filtering, and schema validation checks.

**Why Final Choice Was Better:**  
Ensured reproducible preprocessing across notebook experiments and dashboard deployment.

---

## Prompt 3 — Timeline Story
**Component:** time-series visualization

**Prompt Used:**  
Build a Plotly time-series page showing subreddit activity spikes with rolling averages and automatic spike detection.

**Issue in First Output:**  
The first spike logic was too sensitive and triggered false positives on low-volume subreddits.

**Manual Validation / Fix:**  
Adjusted the rolling baseline window and added minimum-volume thresholds after testing on sparse communities.

**Why Final Choice Was Better:**  
Produced more meaningful narrative spike detection and reduced misleading trend summaries.

---

## Prompt 4 — Narrative Diffusion
**Component:** cross-community spread

**Prompt Used:**  
Create source-to-destination subreddit flow analysis using crossposts and shared URLs.

**Issue in First Output:**  
The initial flow table only counted subreddit totals rather than true source-destination relationships.

**Manual Validation / Fix:**  
Reworked the logic to preserve directional subreddit flow and validated strongest diffusion paths manually.

**Why Final Choice Was Better:**  
This exposed real narrative migration patterns rather than superficial activity counts.

---

## Prompt 5 — Bridge Users
**Component:** network analysis

**Prompt Used:**  
Build a subreddit overlap network based on shared authors and compute centrality metrics.

**Issue in First Output:**  
Deleted users and low-activity accounts distorted overlap centrality.

**Manual Validation / Fix:**  
Removed deleted users, filtered low-signal authors, and stress-tested disconnected graph components.

**Why Final Choice Was Better:**  
The resulting graph better captured true bridge communities and narrative brokers.

---

## Prompt 6 — Topic Clustering
**Component:** embeddings + clustering

**Prompt Used:**  
Cluster Reddit posts using sentence-transformer embeddings, UMAP, and KMeans.

**Issue in First Output:**  
Default cluster count produced overlapping ideological themes.

**Manual Validation / Fix:**  
Tested multiple cluster counts manually and selected the value that best separated major political narratives while preserving interpretability.

**Why Final Choice Was Better:**  
Improved coherence of topic clusters and strengthened downstream narrative migration analysis.

---

## Prompt 7 — Semantic Search
**Component:** retrieval

**Prompt Used:**  
Implement semantic search using sentence-transformers and cosine similarity for zero-keyword-overlap retrieval.

**Issue in First Output:**  
Short queries returned overly broad results.

**Manual Validation / Fix:**  
Added minimum similarity thresholds, fallback handling, and manually validated several cross-topic queries.

**Why Final Choice Was Better:**  
Improved precision and made chatbot exploration more useful for investigative workflows.

---
