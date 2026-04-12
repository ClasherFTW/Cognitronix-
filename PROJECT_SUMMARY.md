# Project Summary: Adaptive Movie-to-Song Recommendation System

## 1) What This Project Does
This project recommends songs for a movie query using an adaptive, content-based pipeline.
It is built for hackathon scenarios where datasets can change and judges may provide new CSV files during demo time.

Instead of relying on user-history collaborative filtering, it learns a shared latent space from movie and song metadata/text.

## 2) Core Features

### A. Adaptive Dataset Handling (Works Beyond One Fixed Dataset)
- Auto-detects key columns like title, id, and artist from candidate names.
- Infers shared feature columns between movies and songs dynamically.
- Automatically classifies shared columns into:
  - Categorical features
  - Numeric features
  - Text features
- Applies robust null handling and fallback behavior.

Why this matters:
- If judges swap in a different dataset schema, the system can still train and run as long as movies and songs share at least one meaningful feature signal.

### B. Adaptive Feature Engineering
- Builds a joint feature space across movies and songs:
  - One-hot encoding for categorical features
  - TF-IDF (with unigrams + bigrams) for text features
  - Scaled numeric features
- Stores feature names for explainability.
- Uses sparse matrices for memory-efficient representation.

### C. Dictionary Learning Latent Space
- Trains Dictionary Learning / MiniBatchDictionaryLearning on the combined feature matrix.
- Produces sparse latent codes for each movie and song.
- Recommendations are driven by similarity in this latent representation.

Why this matters:
- Captures hidden semantic patterns beyond raw metadata matching.
- Keeps model interpretable through atoms/components.

### D. Counterfactual Recommendations
- Supports query-time steering:
  - `--prefer column=value`
  - `--avoid column=value`
  - `--boost_term word`
  - `--cf_strength`
- System shifts the movie latent vector toward preferred prototypes and away from avoided prototypes.
- Adds metadata/text bonus terms to the final ranking score.

Why this matters:
- You can demo controllable AI behavior live, not just static top-N retrieval.

### E. Explainable "Atom DNA"
- `--explain` mode shows why each song was recommended.
- Explanations are generated from top shared latent atoms between movie and song.
- Each atom is mapped to top contributing feature names (e.g., category/text tokens).

Why this matters:
- Judges can see transparent reasons, not black-box outputs.

### F. ANN Retrieval + Exact Fallback
- Uses a lightweight ANN index (random-hyperplane LSH) on song latent codes.
- Fast candidate retrieval for scale.
- Automatically falls back to exact cosine scan when ANN candidate pool is too small.

Why this matters:
- Balances speed and reliability, which is great for real demos.

### G. Diversity-Aware Re-ranking
- Uses MMR-like reranking to reduce repetitive recommendations.
- Keeps recommendations relevant while improving variety.

Why this matters:
- Output quality feels smarter and less redundant.

### H. Cinematic Arc Playlist Generation
- New `arc` mode generates staged playlists for a movie:
  - Setup
  - Conflict
  - Climax
- Scores songs with stage-dependent weighting of similarity, novelty, and energy.

Why this matters:
- Very strong demo storytelling feature, unique and memorable.

### I. Hackathon-Ready Evaluation Metrics
In addition to standard ranking metrics, evaluation now reports:
- Precision@K
- Recall@K
- F1@K
- NDCG@K
- Diversity@K
- Serendipity@K
- Coverage@K

Why this matters:
- Shows both accuracy and experience quality, which is a strong judging advantage.

## 3) How the System Works End-to-End

### Train Flow
1. Load movies and songs CSV.
2. Infer schema and shared features adaptively.
3. Build combined feature matrix.
4. Train dictionary learning model.
5. Save:
   - Model artifacts
   - Sparse codes
   - Feature config/schema
   - ANN index

### Recommend Flow
1. Fuzzy-match input movie title.
2. Load movie latent code and all song codes.
3. Apply optional counterfactual steering.
4. Retrieve candidates via ANN (or exact fallback).
5. Score and optionally diversity-rerank.
6. Return top-N songs with optional atom-level explanations.

### Arc Flow
1. Fuzzy-match movie.
2. Retrieve candidate songs.
3. Compute per-song similarity, novelty, and energy.
4. Select songs stage-by-stage with different narrative profiles.

### Evaluate Flow
1. Hold out test movies.
2. Train on remaining movies + all songs.
3. Recommend for each test movie.
4. Compute both standard and hackathon metrics.

## 4) Main Modules and Responsibilities
- `src/data_loader.py`: adaptive schema detection and robust dataset loading.
- `src/preprocessor.py`: feature-space construction and transforms.
- `src/dictionary_model.py`: dictionary model training and artifact loading.
- `src/ann_index.py`: ANN index build/query.
- `src/recommender.py`: recommendations, counterfactual logic, explanations, arc playlists.
- `src/evaluator.py`: evaluation pipeline and metrics.
- `cli.py`: command-line interface for train/recommend/arc/evaluate.

## 5) Why This Is Hackathon-Strong
- Unique behavior: counterfactual steering + narrative playlist + atom explanations.
- Practical robustness: adaptive schema handling for unseen judge datasets.
- Scalability story: ANN retrieval with fallback.
- Strong evaluation story: quality metrics beyond plain precision/recall.

## 6) Demo Commands (Quick)
- Train: `python cli.py train`
- Recommend: `python cli.py recommend --movie "Inception" --top 5`
- Counterfactual + Explain: `python cli.py recommend --movie "Interstellar" --prefer mood=Peaceful --avoid mood=Dark --boost_term cosmic --explain`
- Arc playlist: `python cli.py arc --movie "Inception" --stages 3 --per_stage 3`
- Evaluate: `python cli.py evaluate`
