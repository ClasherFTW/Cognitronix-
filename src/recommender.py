"""Recommendation and playlist logic using sparse codes from dictionary learning."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity

from src.ann_index import query_ann_index
from src.data_loader import load_datasets
from src.dictionary_model import load_trained_artifacts


def _fuzzy_match_movie(
    query: str,
    movies_df: pd.DataFrame,
    min_score: int = 55,
) -> Dict[str, Optional[object]]:
    """Find the closest movie title using fuzzy matching and provide suggestions."""
    choices = movies_df["title"].tolist()
    matches = process.extract(query, choices, scorer=fuzz.WRatio, limit=5)
    if not matches:
        return {"matched": None, "suggestions": []}

    best_title, best_score, best_index = matches[0]
    suggestions = [title for title, _, _ in matches]
    if best_score < min_score:
        return {"matched": None, "suggestions": suggestions}

    return {
        "matched": {"title": best_title, "score": float(best_score), "index": int(best_index)},
        "suggestions": suggestions,
    }


def _prepare_constraints(constraints: Optional[Dict[str, Iterable[str]]]) -> Dict[str, List[str]]:
    """Normalize counterfactual constraints to lowercase value lists."""
    prepared: Dict[str, List[str]] = {}
    if not constraints:
        return prepared
    for column, values in constraints.items():
        if values is None:
            continue
        normalized = sorted({str(value).strip().lower() for value in values if str(value).strip()})
        if normalized:
            prepared[str(column).strip()] = normalized
    return prepared


def _constraint_match_scores(songs_df: pd.DataFrame, constraints: Dict[str, List[str]]) -> np.ndarray:
    """Compute per-song match counts for constraints."""
    if not constraints:
        return np.zeros(len(songs_df), dtype=np.float32)

    scores = np.zeros(len(songs_df), dtype=np.float32)
    column_lookup = {column.lower(): column for column in songs_df.columns}
    for column, values in constraints.items():
        resolved = column if column in songs_df.columns else column_lookup.get(column.lower())
        if resolved is None:
            continue
        matches = songs_df[resolved].astype(str).str.lower().isin(values).to_numpy(dtype=np.float32)
        scores += matches
    return scores


def _prototype_from_constraints(
    song_vectors: np.ndarray,
    songs_df: pd.DataFrame,
    constraints: Dict[str, List[str]],
) -> Optional[np.ndarray]:
    """Create a latent prototype vector from constraint-matching songs."""
    scores = _constraint_match_scores(songs_df, constraints)
    if float(scores.max(initial=0.0)) <= 0.0:
        return None
    weights = scores / float(scores.sum())
    return (weights.reshape(-1, 1) * song_vectors).sum(axis=0)


def _apply_counterfactual_shift(
    movie_vector: np.ndarray,
    song_vectors: np.ndarray,
    songs_df: pd.DataFrame,
    prefer: Dict[str, List[str]],
    avoid: Dict[str, List[str]],
    strength: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift query vector toward preferred prototypes and away from avoided prototypes."""
    adjusted = movie_vector.astype(np.float32).copy()
    prefer_scores = _constraint_match_scores(songs_df, prefer)
    avoid_scores = _constraint_match_scores(songs_df, avoid)
    prefer_proto = _prototype_from_constraints(song_vectors, songs_df, prefer)
    avoid_proto = _prototype_from_constraints(song_vectors, songs_df, avoid)

    if prefer_proto is not None:
        adjusted += float(strength) * prefer_proto
    if avoid_proto is not None:
        adjusted -= float(strength) * avoid_proto

    norm = np.linalg.norm(adjusted)
    if norm > 0.0:
        adjusted = adjusted / norm
    return adjusted, prefer_scores, avoid_scores


def _metadata_bonus(
    songs_df: pd.DataFrame,
    prefer_scores: np.ndarray,
    avoid_scores: np.ndarray,
    boost_terms: Optional[List[str]],
    text_columns: List[str],
    prefer_weight: float = 0.08,
    avoid_weight: float = 0.10,
    term_weight: float = 0.06,
) -> np.ndarray:
    """Compute metadata-aware score bonus for re-ranking."""
    bonus = (prefer_weight * prefer_scores) - (avoid_weight * avoid_scores)
    if not boost_terms:
        return bonus.astype(np.float32)

    normalized_terms = [term.strip().lower() for term in boost_terms if term.strip()]
    if not normalized_terms or not text_columns:
        return bonus.astype(np.float32)

    text_blob = pd.Series([""] * len(songs_df), index=songs_df.index, dtype=str)
    for column in text_columns:
        if column in songs_df.columns:
            text_blob = text_blob + " " + songs_df[column].fillna("").astype(str)
    text_blob = text_blob.str.lower()
    term_hits = np.zeros(len(songs_df), dtype=np.float32)
    for term in normalized_terms:
        term_hits += text_blob.str.contains(term, regex=False).to_numpy(dtype=np.float32)
    return (bonus + (term_weight * term_hits)).astype(np.float32)


def _mmr_select(
    candidate_indices: np.ndarray,
    relevance_scores: np.ndarray,
    song_vectors: np.ndarray,
    top_n: int,
    diversity_lambda: float = 0.78,
) -> np.ndarray:
    """Select top-N with Maximum Marginal Relevance to improve diversity."""
    if len(candidate_indices) <= top_n:
        return candidate_indices

    selected_positions: List[int] = []
    remaining_positions = list(range(len(candidate_indices)))
    vectors = song_vectors[candidate_indices]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized_vectors = vectors / norms

    while remaining_positions and len(selected_positions) < top_n:
        if not selected_positions:
            best_pos = max(remaining_positions, key=lambda pos: relevance_scores[pos])
            selected_positions.append(best_pos)
            remaining_positions.remove(best_pos)
            continue

        def objective(pos: int) -> float:
            candidate_vec = normalized_vectors[pos].reshape(1, -1)
            selected_vecs = normalized_vectors[selected_positions]
            diversity_penalty = float(np.max(candidate_vec @ selected_vecs.T))
            return float(diversity_lambda * relevance_scores[pos] - (1.0 - diversity_lambda) * diversity_penalty)

        best_pos = max(remaining_positions, key=objective)
        selected_positions.append(best_pos)
        remaining_positions.remove(best_pos)

    return candidate_indices[np.asarray(selected_positions, dtype=np.int32)]


def _build_atom_explanation(
    movie_code: np.ndarray,
    song_code: np.ndarray,
    dictionary: np.ndarray,
    feature_names: List[str],
    top_atoms: int = 3,
    top_features_per_atom: int = 3,
) -> str:
    """Create a compact latent-atom explanation string for a movie-song pair."""
    shared_strength = np.abs(movie_code) * np.abs(song_code)
    if shared_strength.size == 0:
        return "No shared latent atoms."

    atom_indices = np.argsort(shared_strength)[::-1][:top_atoms]
    explanation_parts: List[str] = []
    for atom_idx in atom_indices.tolist():
        if atom_idx >= dictionary.shape[0]:
            continue
        atom_weights = np.abs(dictionary[atom_idx])
        feature_idx = np.argsort(atom_weights)[::-1][:top_features_per_atom]
        atom_features = [feature_names[idx] for idx in feature_idx.tolist() if idx < len(feature_names)]
        if atom_features:
            explanation_parts.append(f"A{atom_idx}:{', '.join(atom_features)}")
    return " | ".join(explanation_parts) if explanation_parts else "No strong feature atoms."


def _runtime_context(
    data_dir: str,
    model_dir: str,
    movies_filename: str = "movies.csv",
    songs_filename: str = "songs.csv",
    max_cross_songs: int = 15000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, object], Dict[str, Any]]:
    """Load data and model artifacts and validate boundary compatibility."""
    movies_df, songs_df, schema = load_datasets(
        data_dir=data_dir,
        movies_filename=movies_filename,
        songs_filename=songs_filename,
        max_cross_songs=max_cross_songs,
    )
    artifacts = load_trained_artifacts(model_dir=model_dir)
    boundaries = artifacts["boundaries"]
    if (boundaries["movie_end"] - boundaries["movie_start"]) != len(movies_df):
        raise ValueError("The current movie dataset differs from training artifacts. Retrain with `python cli.py train`.")
    if (boundaries["song_end"] - boundaries["song_start"]) != len(songs_df):
        raise ValueError("The current dataset differs from training artifacts. Retrain with `python cli.py train`.")

    preprocess_artifacts = {
        "encoder": artifacts.get("encoder"),
        "vectorizer": artifacts.get("vectorizer"),
        "scaler": artifacts.get("scaler"),
        "feature_config": artifacts.get(
            "feature_config",
            {
                "categorical_columns": schema.get("categorical_columns", []),
                "numeric_columns": schema.get("numeric_columns", []),
                "text_columns": schema.get("text_columns", []),
            },
        ),
        "feature_names": artifacts.get("feature_names", []),
        "schema": artifacts.get("schema", schema),
    }
    return movies_df, songs_df, artifacts, boundaries, preprocess_artifacts


def recommend_songs(
    movie_title: str,
    top_n: int = 5,
    data_dir: str = "data",
    model_dir: str = "models",
    movies_filename: str = "movies.csv",
    songs_filename: str = "songs.csv",
    max_cross_songs: int = 15000,
    min_match_score: int = 55,
    prefer: Optional[Dict[str, Iterable[str]]] = None,
    avoid: Optional[Dict[str, Iterable[str]]] = None,
    boost_terms: Optional[List[str]] = None,
    counterfactual_strength: float = 0.35,
    explain: bool = False,
    use_ann: bool = True,
    ann_search_k: int = 256,
    ann_hamming_radius: int = 2,
    diversify: bool = True,
) -> Dict[str, object]:
    """Recommend top-N songs for a movie with counterfactual controls and ANN retrieval."""
    movies_df, songs_df, artifacts, boundaries, preprocess_artifacts = _runtime_context(
        data_dir=data_dir,
        model_dir=model_dir,
        movies_filename=movies_filename,
        songs_filename=songs_filename,
        max_cross_songs=max_cross_songs,
    )
    sparse_codes = np.asarray(artifacts["sparse_codes"], dtype=np.float32)
    dictionary = np.asarray(artifacts["dictionary"], dtype=np.float32)
    feature_names = preprocess_artifacts.get("feature_names", [])
    text_columns = list(preprocess_artifacts["feature_config"].get("text_columns", []))

    movie_match = _fuzzy_match_movie(movie_title, movies_df, min_score=min_match_score)
    if movie_match["matched"] is None:
        return {"matched_movie": None, "suggestions": movie_match["suggestions"], "results": pd.DataFrame()}

    local_movie_idx = int(movie_match["matched"]["index"])
    movie_global_idx = boundaries["movie_start"] + local_movie_idx
    song_start = boundaries["song_start"]
    song_end = boundaries["song_end"]

    movie_vector = sparse_codes[movie_global_idx].astype(np.float32)
    song_vectors = sparse_codes[song_start:song_end].astype(np.float32)
    prepared_prefer = _prepare_constraints(prefer)
    prepared_avoid = _prepare_constraints(avoid)

    adjusted_query, prefer_scores, avoid_scores = _apply_counterfactual_shift(
        movie_vector=movie_vector,
        song_vectors=song_vectors,
        songs_df=songs_df,
        prefer=prepared_prefer,
        avoid=prepared_avoid,
        strength=counterfactual_strength,
    )
    bonus_scores = _metadata_bonus(
        songs_df=songs_df,
        prefer_scores=prefer_scores,
        avoid_scores=avoid_scores,
        boost_terms=boost_terms,
        text_columns=text_columns,
    )

    retrieval_method = "exact"
    if use_ann and artifacts.get("ann_index") is not None:
        retrieval_method = "ann"
        ann_top_k = max(int(ann_search_k), top_n * 30)
        candidate_indices, ann_similarities, candidate_count = query_ann_index(
            query_vector=adjusted_query,
            ann_index=artifacts["ann_index"],
            top_n=ann_top_k,
            search_k=max(int(ann_search_k), ann_top_k),
            max_hamming_radius=ann_hamming_radius,
        )
        if candidate_count < max(top_n * 5, top_n):
            candidate_indices = np.arange(len(song_vectors), dtype=np.int32)
            candidate_base = cosine_similarity(adjusted_query.reshape(1, -1), song_vectors).ravel().astype(np.float32)
            candidate_count = int(len(candidate_indices))
            retrieval_method = "exact_fallback"
        else:
            candidate_base = ann_similarities.astype(np.float32)
    else:
        candidate_indices = np.arange(len(song_vectors), dtype=np.int32)
        candidate_base = cosine_similarity(adjusted_query.reshape(1, -1), song_vectors).ravel().astype(np.float32)
        candidate_count = int(len(candidate_indices))

    candidate_bonus = bonus_scores[candidate_indices]
    candidate_scores = candidate_base + candidate_bonus
    ranked_pos = np.argsort(candidate_scores)[::-1]
    candidate_indices = candidate_indices[ranked_pos]
    candidate_scores = candidate_scores[ranked_pos]
    candidate_base = candidate_base[ranked_pos]

    if diversify:
        selected_indices = _mmr_select(
            candidate_indices=candidate_indices[: max(top_n * 10, top_n)],
            relevance_scores=candidate_scores[: max(top_n * 10, top_n)],
            song_vectors=song_vectors,
            top_n=top_n,
            diversity_lambda=0.78,
        )
        final_indices = np.asarray(selected_indices, dtype=np.int32)
    else:
        final_indices = candidate_indices[:top_n]

    base_similarity_map = {int(idx): float(val) for idx, val in zip(candidate_indices.tolist(), candidate_base.tolist())}
    final_score_map = {int(idx): float(val) for idx, val in zip(candidate_indices.tolist(), candidate_scores.tolist())}
    recommendations = songs_df.iloc[final_indices][["title", "artist"]].copy()
    recommendations.insert(0, "rank", np.arange(1, len(recommendations) + 1))
    recommendations["similarity"] = recommendations.index.map(
        lambda original_idx: base_similarity_map.get(int(original_idx), 0.0)
    )
    recommendations["score"] = recommendations.index.map(
        lambda original_idx: final_score_map.get(int(original_idx), 0.0)
    )
    recommendations["method"] = retrieval_method
    recommendations = recommendations.reset_index(drop=True)

    if explain:
        explanations: List[str] = []
        for idx in final_indices.tolist():
            song_code = song_vectors[idx]
            explanation = _build_atom_explanation(
                movie_code=movie_vector,
                song_code=song_code,
                dictionary=dictionary,
                feature_names=feature_names,
            )
            explanations.append(explanation)
        recommendations["explanation"] = explanations

    return {
        "matched_movie": movie_match["matched"],
        "suggestions": movie_match["suggestions"],
        "results": recommendations,
        "candidate_pool_size": candidate_count,
    }


def _energy_signal(songs_df: pd.DataFrame, song_vectors: np.ndarray) -> np.ndarray:
    """Estimate intensity/energy signal for arc playlist generation."""
    if "energy_level" in songs_df.columns:
        mapping = {"low": 0.2, "medium": 0.6, "high": 1.0}
        return (
            songs_df["energy_level"]
            .astype(str)
            .str.lower()
            .map(mapping)
            .fillna(0.5)
            .to_numpy(dtype=np.float32)
        )

    for column in songs_df.columns:
        col_lower = column.lower()
        if "energy" in col_lower or "intensity" in col_lower or "tempo" in col_lower:
            values = pd.to_numeric(songs_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            max_value = float(np.max(values)) if len(values) > 0 else 0.0
            min_value = float(np.min(values)) if len(values) > 0 else 0.0
            if max_value - min_value > 1e-8:
                return (values - min_value) / (max_value - min_value)

    norms = np.linalg.norm(song_vectors, axis=1).astype(np.float32)
    max_norm = float(np.max(norms)) if len(norms) > 0 else 0.0
    min_norm = float(np.min(norms)) if len(norms) > 0 else 0.0
    if max_norm - min_norm > 1e-8:
        return (norms - min_norm) / (max_norm - min_norm)
    return np.zeros(len(song_vectors), dtype=np.float32)


def _arc_profiles(num_stages: int) -> List[Tuple[str, float, float, float]]:
    """Create stage profiles of (name, similarity_weight, novelty_weight, energy_weight)."""
    base = [
        ("setup", 0.80, 0.15, 0.05),
        ("conflict", 0.58, 0.27, 0.15),
        ("climax", 0.40, 0.25, 0.35),
    ]
    if num_stages <= len(base):
        return base[:num_stages]

    profiles = []
    for index in range(num_stages):
        t = index / max(num_stages - 1, 1)
        sim_w = 0.82 - (0.42 * t)
        novelty_w = 0.12 + (0.18 * t)
        energy_w = 1.0 - sim_w - novelty_w
        profiles.append((f"stage_{index + 1}", sim_w, novelty_w, energy_w))
    return profiles


def generate_arc_playlist(
    movie_title: str,
    stages: int = 3,
    per_stage: int = 3,
    data_dir: str = "data",
    model_dir: str = "models",
    movies_filename: str = "movies.csv",
    songs_filename: str = "songs.csv",
    max_cross_songs: int = 15000,
    min_match_score: int = 55,
    use_ann: bool = True,
    ann_search_k: int = 512,
) -> Dict[str, object]:
    """Generate a multi-stage cinematic arc playlist for a movie."""
    movies_df, songs_df, artifacts, boundaries, _ = _runtime_context(
        data_dir=data_dir,
        model_dir=model_dir,
        movies_filename=movies_filename,
        songs_filename=songs_filename,
        max_cross_songs=max_cross_songs,
    )
    sparse_codes = np.asarray(artifacts["sparse_codes"], dtype=np.float32)

    movie_match = _fuzzy_match_movie(movie_title, movies_df, min_score=min_match_score)
    if movie_match["matched"] is None:
        return {"matched_movie": None, "suggestions": movie_match["suggestions"], "results": pd.DataFrame()}

    movie_idx = boundaries["movie_start"] + int(movie_match["matched"]["index"])
    song_start = boundaries["song_start"]
    song_end = boundaries["song_end"]
    movie_vector = sparse_codes[movie_idx]
    song_vectors = sparse_codes[song_start:song_end]

    if use_ann and artifacts.get("ann_index") is not None:
        candidate_indices, _, _ = query_ann_index(
            query_vector=movie_vector,
            ann_index=artifacts["ann_index"],
            top_n=max(ann_search_k, stages * per_stage * 20),
            search_k=max(ann_search_k, stages * per_stage * 20),
            max_hamming_radius=2,
        )
        if len(candidate_indices) < max(stages * per_stage * 5, stages * per_stage):
            candidate_indices = np.arange(len(song_vectors), dtype=np.int32)
    else:
        candidate_indices = np.arange(len(song_vectors), dtype=np.int32)

    candidate_vectors = song_vectors[candidate_indices]
    similarities = cosine_similarity(movie_vector.reshape(1, -1), candidate_vectors).ravel().astype(np.float32)
    novelty = (1.0 - ((similarities + 1.0) / 2.0)).astype(np.float32)
    energy = _energy_signal(songs_df.iloc[candidate_indices].reset_index(drop=True), candidate_vectors)
    profiles = _arc_profiles(max(1, stages))

    used = set()
    rows: List[Dict[str, object]] = []
    for stage_idx, (stage_name, sim_w, novelty_w, energy_w) in enumerate(profiles, start=1):
        stage_scores = (sim_w * similarities) + (novelty_w * novelty) + (energy_w * energy)
        order = np.argsort(stage_scores)[::-1]
        picked = 0
        for pos in order.tolist():
            global_song_idx = int(candidate_indices[pos])
            if global_song_idx in used:
                continue
            used.add(global_song_idx)
            song_row = songs_df.iloc[global_song_idx]
            picked += 1
            rows.append(
                {
                    "stage": stage_name,
                    "stage_rank": picked,
                    "title": song_row["title"],
                    "artist": song_row.get("artist", "Unknown Artist"),
                    "arc_score": float(stage_scores[pos]),
                    "similarity": float(similarities[pos]),
                    "novelty": float(novelty[pos]),
                    "energy": float(energy[pos]),
                }
            )
            if picked >= per_stage:
                break

    results = pd.DataFrame(rows)
    return {
        "matched_movie": movie_match["matched"],
        "suggestions": movie_match["suggestions"],
        "results": results,
    }


def format_recommendations_table(recommendations: pd.DataFrame) -> str:
    """Format recommendation results as a readable CLI table."""
    if recommendations.empty:
        return "No recommendations available."

    printable = recommendations.copy()
    for column in ["similarity", "score", "arc_score", "novelty", "energy"]:
        if column in printable.columns:
            printable[column] = printable[column].map(lambda value: f"{value:.4f}")
    return printable.to_string(index=False)
