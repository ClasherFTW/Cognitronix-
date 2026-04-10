# Dictionary Learning Movie-to-Song Recommendation System

## Overview
This project builds a CLI-based recommendation engine that maps movies to compatible songs using:
- A shared content feature space (categorical metadata + TF-IDF tags)
- `DictionaryLearning` from scikit-learn to learn latent atoms
- Sparse representations (codes) for both movies and songs
- Cosine similarity in sparse-code space for retrieval

It is a pure content-based system and does not use collaborative filtering.

## Project Structure
```text
recommendation_system/
├── data/
│   ├── movies.csv
│   └── songs.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── dictionary_model.py
│   ├── recommender.py
│   └── evaluator.py
├── models/
├── cli.py
├── requirements.txt
└── README.md
```

## Setup
1. Use Python 3.9+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train
```bash
python cli.py train
```

Custom hyperparameters:
```bash
python cli.py train --n_components 30 --alpha 0.5
```

Training prints:
- Number of total items
- Feature dimensionality
- Sparse-code sparsity percentage
- Total time taken

## Recommend
```bash
python cli.py recommend --movie "Interstellar"
```

Top-N results:
```bash
python cli.py recommend --movie "Inception" --top 5
```

Movie lookup uses fuzzy matching (`rapidfuzz`), so partial or misspelled names are supported.

## Evaluate
```bash
python cli.py evaluate
```

Evaluation protocol:
- Hold out 20% movies as test movies
- Train on remaining movies + all songs
- Define relevance as songs matching at least 2 attributes among:
  `genre`, `mood`, `era`, `language`, `energy_level`
- Report:
  - Precision@K
  - Recall@K
  - F1@K
  - NDCG@K
- Metrics are printed for `K=5` and `K=10`

## Why This Handles Cold Start
Because recommendations are content-based, only metadata and tags are required. New movies or songs can be represented immediately in the learned feature space without user interaction history.

## Key Hyperparameters
- `n_components`: Number of dictionary atoms.  
  Higher values can capture more nuanced patterns but may overfit and increase compute.
- `alpha`: Sparsity penalty in L1 coding.  
  Higher alpha enforces sparser representations; lower alpha allows denser codes.

Suggested tuning strategy:
1. Start with `n_components=20`, `alpha=1.0`.
2. Increase `n_components` gradually (e.g., 20 -> 30 -> 40).
3. Sweep `alpha` (e.g., 0.5, 1.0, 1.5).
4. Select the setting with better `NDCG@10` and `F1@10`.
