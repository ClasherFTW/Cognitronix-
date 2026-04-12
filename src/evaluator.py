"""Evaluation utilities for the adaptive dictionary-learning recommender."""

import math
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.data_loader import load_datasets
from src.dictionary_model import train_dictionary_model
from src.preprocessor import fit_feature_space, transform_with_feature_space


def _split_movies(
    movies_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split movies into train and test partitions."""
    indices = np.arange(len(movies_df))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, shuffle=True
    )
    train_movies = movies_df.iloc[train_indices].reset_index(drop=True)
    test_movies = movies_df.iloc[test_indices].reset_index(drop=True)
    return train_movies, test_movies


def _tokenize(text: str) -> Set[str]:
    """Tokenize lightweight text to a set of lowercase tokens."""
    if not text:
        return set()
    return {token.strip().lower() for token in str(text).replace(",", " ").split() if token.strip()}


def _relevant_song_indices(
    movie_row: pd.Series,
    songs_df: pd.DataFrame,
    schema: Dict[str, Any],
) -> Set[int]:
    """Return relevant song indices using adaptive schema-aware matching."""
    categorical_columns = list(schema.get("categorical_columns", []))
    numeric_columns = list(schema.get("numeric_columns", []))
    text_columns = list(schema.get("text_columns", []))

    relevance = np.zeros(len(songs_df), dtype=np.float32)
    total_signals = 0

    for column in categorical_columns:
        if column not in songs_df.columns or column not in movie_row.index:
            continue
        total_signals += 1
        relevance += (songs_df[column].astype(str) == str(movie_row[column])).to_numpy(dtype=np.float32)

    for column in numeric_columns:
        if column not in songs_df.columns or column not in movie_row.index:
            continue
        total_signals += 1
        song_values = pd.to_numeric(songs_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        movie_value = float(pd.to_numeric(pd.Series([movie_row[column]]), errors="coerce").fillna(0.0).iloc[0])
        tolerance = max(float(np.nanstd(song_values)) * 0.35, 1e-4)
        relevance += (np.abs(song_values - movie_value) <= tolerance).astype(np.float32)

    for column in text_columns:
        if column not in songs_df.columns or column not in movie_row.index:
            continue
        total_signals += 1
        movie_tokens = _tokenize(movie_row[column])
        if not movie_tokens:
            continue
        overlaps = songs_df[column].fillna("").astype(str).map(
            lambda value: len(movie_tokens.intersection(_tokenize(value)))
        )
        relevance += (overlaps > 0).to_numpy(dtype=np.float32)

    if total_signals == 0:
        return set()

    min_required = max(1, int(math.ceil(0.35 * total_signals)))
    relevant = np.where(relevance >= min_required)[0].tolist()
    if not relevant:
        best = np.argsort(relevance)[::-1][: max(1, min(10, len(songs_df)))]
        return set(best.tolist())
    return set(relevant)


def _precision_recall_f1_at_k(
    ranked_indices: List[int], relevant_indices: Set[int], k: int
) -> Tuple[float, float, float]:
    """Compute Precision@K, Recall@K, and F1@K for one query item."""
    top_k = ranked_indices[:k]
    hits = sum(1 for idx in top_k if idx in relevant_indices)

    precision = hits / float(k) if k > 0 else 0.0
    recall = hits / float(len(relevant_indices)) if relevant_indices else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _ndcg_at_k(ranked_indices: List[int], relevant_indices: Set[int], k: int) -> float:
    """Compute binary NDCG@K for a ranked recommendation list."""
    top_k = ranked_indices[:k]
    dcg = 0.0
    for rank, idx in enumerate(top_k):
        if idx in relevant_indices:
            dcg += 1.0 / np.log2(rank + 2.0)

    ideal_hits = min(len(relevant_indices), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 2.0) for rank in range(ideal_hits))
    return dcg / idcg


def _diversity_at_k(ranked_indices: List[int], song_codes: np.ndarray, k: int) -> float:
    """Compute intra-list diversity at K using average pairwise cosine distance."""
    top_k = ranked_indices[:k]
    if len(top_k) < 2:
        return 0.0
    vectors = song_codes[top_k]
    sim_matrix = cosine_similarity(vectors, vectors)
    upper_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sim = sim_matrix[upper_indices]
    return float(np.mean(1.0 - pairwise_sim)) if pairwise_sim.size > 0 else 0.0


def _serendipity_at_k(
    ranked_indices: List[int],
    relevant_indices: Set[int],
    similarities: np.ndarray,
    k: int,
) -> float:
    """Compute serendipity as relevant-yet-unexpected recommendations at K."""
    top_k = ranked_indices[:k]
    hit_indices = [idx for idx in top_k if idx in relevant_indices]
    if not hit_indices:
        return 0.0
    hit_sim = similarities[hit_indices]
    normalized = (hit_sim + 1.0) / 2.0
    unexpectedness = 1.0 - normalized
    return float(np.mean(unexpectedness))


def evaluate_recommender(
    data_dir: str = "data",
    model_dir: str = "models",
    movies_filename: str = "movies.csv",
    songs_filename: str = "songs.csv",
    max_cross_songs: int = 15000,
    n_components: int = 20,
    alpha: float = 1.0,
    k_values: Iterable[int] = (5, 10),
    test_size: float = 0.2,
    random_state: int = 42,
    max_text_features: int = 3000,
    min_text_df: int = 1,
) -> Tuple[pd.DataFrame, int]:
    """Evaluate recommendations with adaptive relevance and hackathon-ready metrics."""
    movies_df, songs_df, schema = load_datasets(
        data_dir=data_dir,
        movies_filename=movies_filename,
        songs_filename=songs_filename,
        max_cross_songs=max_cross_songs,
    )
    train_movies, test_movies = _split_movies(movies_df, test_size=test_size, random_state=random_state)

    train_matrix, boundaries, preprocess_artifacts = fit_feature_space(
        train_movies,
        songs_df,
        schema=schema,
        model_dir=model_dir,
        save_artifacts=False,
        max_text_features=max_text_features,
        min_text_df=min_text_df,
    )
    trained = train_dictionary_model(
        train_matrix,
        n_components=n_components,
        alpha=alpha,
        random_state=random_state,
        use_minibatch=True,
        model_dir=model_dir,
        save_artifacts=False,
    )
    dictionary_model = trained["model"]
    train_sparse_codes = np.asarray(trained["sparse_codes"], dtype=np.float32)
    song_codes = train_sparse_codes[boundaries["song_start"] : boundaries["song_end"]]

    test_matrix = transform_with_feature_space(test_movies, preprocess_artifacts)
    test_codes = dictionary_model.transform(test_matrix.toarray()).astype(np.float32)

    k_values_list = list(k_values)
    metric_buckets: Dict[int, Dict[str, List[float]]] = {
        k: {
            "precision": [],
            "recall": [],
            "f1": [],
            "ndcg": [],
            "diversity": [],
            "serendipity": [],
        }
        for k in k_values_list
    }
    coverage_sets: Dict[int, Set[int]] = {k: set() for k in k_values_list}

    for movie_idx, movie_row in test_movies.iterrows():
        movie_code = test_codes[movie_idx].reshape(1, -1)
        similarities = cosine_similarity(movie_code, song_codes).ravel().astype(np.float32)
        ranked_song_indices = np.argsort(similarities)[::-1].tolist()
        relevant = _relevant_song_indices(movie_row, songs_df, schema)

        for k in k_values_list:
            precision, recall, f1 = _precision_recall_f1_at_k(ranked_song_indices, relevant, k)
            ndcg = _ndcg_at_k(ranked_song_indices, relevant, k)
            diversity = _diversity_at_k(ranked_song_indices, song_codes, k)
            serendipity = _serendipity_at_k(ranked_song_indices, relevant, similarities, k)

            metric_buckets[k]["precision"].append(precision)
            metric_buckets[k]["recall"].append(recall)
            metric_buckets[k]["f1"].append(f1)
            metric_buckets[k]["ndcg"].append(ndcg)
            metric_buckets[k]["diversity"].append(diversity)
            metric_buckets[k]["serendipity"].append(serendipity)
            coverage_sets[k].update(ranked_song_indices[:k])

    rows = []
    total_songs = max(len(songs_df), 1)
    for k in k_values_list:
        rows.append(
            {
                "K": k,
                "Precision@K": float(np.mean(metric_buckets[k]["precision"])) if metric_buckets[k]["precision"] else 0.0,
                "Recall@K": float(np.mean(metric_buckets[k]["recall"])) if metric_buckets[k]["recall"] else 0.0,
                "F1@K": float(np.mean(metric_buckets[k]["f1"])) if metric_buckets[k]["f1"] else 0.0,
                "NDCG@K": float(np.mean(metric_buckets[k]["ndcg"])) if metric_buckets[k]["ndcg"] else 0.0,
                "Diversity@K": float(np.mean(metric_buckets[k]["diversity"])) if metric_buckets[k]["diversity"] else 0.0,
                "Serendipity@K": float(np.mean(metric_buckets[k]["serendipity"]))
                if metric_buckets[k]["serendipity"]
                else 0.0,
                "Coverage@K": float(len(coverage_sets[k]) / total_songs),
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df, len(test_movies)
