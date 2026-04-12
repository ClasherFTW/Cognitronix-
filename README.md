# Adaptive Dictionary-Learning Movie-to-Song Recommender

## Overview
This project is a content-based recommendation engine that maps movies to songs using:
- Adaptive feature inference from any compatible movie/song CSV schema
- Sparse latent representations via dictionary learning
- Counterfactual query steering (prefer/avoid/boost controls)
- Explainable latent atom "DNA" reasoning
- Cinematic arc playlist generation
- ANN retrieval (random-hyperplane LSH) with exact-search fallback

It is designed for hackathon demos where datasets may change on the fly.

## Project Structure
```text
recommendation_system/
|-- data/
|   |-- movies.csv (optional direct mode)
|   |-- songs.csv  (optional direct mode)
|   |-- movielens/
|   `-- last fm/
|-- models/
|-- src/
|   |-- ann_index.py
|   |-- data_loader.py
|   |-- dictionary_model.py
|   |-- evaluator.py
|   |-- preprocessor.py
|   `-- recommender.py
|-- cli.py
|-- requirements.txt
`-- README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python cli.py train --max_cross_songs 5000 --max_text_features 2500 --n_components 32 --alpha 0.25
```

Optional:
```bash
python cli.py train --max_cross_songs 15000 --max_text_features 3000 --n_components 40 --alpha 0.25
```

## Recommend (Counterfactual + Explainable)
Basic:
```bash
python cli.py recommend --movie "Toy Story" --top 5 --max_cross_songs 5000
```

Counterfactual controls:
```bash
python cli.py recommend --movie "Toy Story" --max_cross_songs 5000 --prefer era=2010s --avoid genre=Unknown --boost_term love
```

Explainable latent atoms:
```bash
python cli.py recommend --movie "Toy Story" --max_cross_songs 5000 --explain
```

## Cinematic Arc Playlist
```bash
python cli.py arc --movie "Toy Story" --stages 3 --per_stage 3 --max_cross_songs 5000
```

## Evaluate (Hackathon Metrics)
```bash
python cli.py evaluate --max_cross_songs 5000 --max_text_features 2500 --n_components 32 --alpha 0.25
```

Metrics reported:
- Precision@K
- Recall@K
- F1@K
- NDCG@K
- Diversity@K
- Serendipity@K
- Coverage@K

## Dataset Flexibility
The loader auto-detects:
- Movie and song title columns
- ID columns
- Artist column (optional in songs)
- Shared categorical / numeric / text feature columns

As long as both files share at least one useful feature signal, the pipeline can train and recommend without hardcoded domain vocabulary.

Special support:
- If `data/movies.csv` + `data/songs.csv` exist, they are used directly.
- Otherwise, the system auto-bridges:
  - `data/movielens/*.csv` as movie source
  - `data/last fm/*.csv` as song source
  - and builds a unified trainable schema automatically.

## Notes on ANN
- Retrieval uses ANN index by default.
- If ANN candidate pool is too small, the system auto-falls back to exact cosine scan for stability.
