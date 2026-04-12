"""Lightweight ANN index based on random-hyperplane LSH for cosine retrieval."""

from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize matrix rows for cosine retrieval."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _signatures_to_keys(signatures: np.ndarray) -> np.ndarray:
    """Pack boolean signatures into uint64 keys."""
    n_bits = signatures.shape[1]
    bit_weights = (1 << np.arange(n_bits, dtype=np.uint64)).reshape(1, -1)
    return (signatures.astype(np.uint64) * bit_weights).sum(axis=1).astype(np.uint64)


def _build_buckets(keys: np.ndarray) -> Dict[int, np.ndarray]:
    """Build bucket map from hash key to item indices."""
    buckets: Dict[int, List[int]] = {}
    for index, key in enumerate(keys.tolist()):
        buckets.setdefault(int(key), []).append(index)
    return {key: np.asarray(indices, dtype=np.int32) for key, indices in buckets.items()}


def build_ann_index(
    song_codes: np.ndarray,
    n_bits: int = 24,
    random_state: int = 42,
    model_dir: str = "models",
    save_artifacts: bool = True,
) -> Dict[str, object]:
    """Build and optionally persist an ANN index for song codes."""
    if n_bits <= 0 or n_bits > 63:
        raise ValueError("n_bits must be in range [1, 63] for uint64 key packing.")

    song_codes = np.asarray(song_codes, dtype=np.float32)
    normalized_song_codes = _normalize_rows(song_codes)
    rng = np.random.default_rng(random_state)
    hyperplanes = rng.standard_normal((n_bits, normalized_song_codes.shape[1])).astype(np.float32)
    signatures = (normalized_song_codes @ hyperplanes.T) >= 0.0
    keys = _signatures_to_keys(signatures)
    buckets = _build_buckets(keys)

    index = {
        "n_bits": int(n_bits),
        "hyperplanes": hyperplanes,
        "keys": keys,
        "buckets": buckets,
        "normalized_song_codes": normalized_song_codes,
    }
    if save_artifacts:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(index, model_path / "ann_index.joblib")
    return index


def _neighbor_keys(base_key: int, n_bits: int, radius: int) -> Iterable[int]:
    """Yield hash keys at Hamming distance up to `radius` from `base_key`."""
    yield base_key
    bit_positions = list(range(n_bits))
    for distance in range(1, radius + 1):
        for bits in combinations(bit_positions, distance):
            neighbor = base_key
            for bit in bits:
                neighbor ^= 1 << bit
            yield neighbor


def query_ann_index(
    query_vector: np.ndarray,
    ann_index: Dict[str, object],
    top_n: int = 5,
    search_k: int = 256,
    max_hamming_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Query ANN index and return top song indices, similarities, and candidate count."""
    n_bits = int(ann_index["n_bits"])
    hyperplanes = np.asarray(ann_index["hyperplanes"], dtype=np.float32)
    buckets = ann_index["buckets"]
    normalized_song_codes = np.asarray(ann_index["normalized_song_codes"], dtype=np.float32)

    query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    normalized_query = _normalize_rows(query_vector).ravel()
    signature = (normalized_query.reshape(1, -1) @ hyperplanes.T) >= 0.0
    base_key = int(_signatures_to_keys(signature)[0])

    candidates: List[int] = []
    seen = set()
    for radius in range(max_hamming_radius + 1):
        for key in _neighbor_keys(base_key, n_bits, radius):
            if key in seen:
                continue
            seen.add(key)
            bucket_indices = buckets.get(key)
            if bucket_indices is None:
                continue
            candidates.extend(bucket_indices.tolist())
        if len(candidates) >= search_k:
            break

    if not candidates:
        candidate_indices = np.arange(normalized_song_codes.shape[0], dtype=np.int32)
    else:
        candidate_indices = np.unique(np.asarray(candidates, dtype=np.int32))

    candidate_vectors = normalized_song_codes[candidate_indices]
    similarities = candidate_vectors @ normalized_query
    order = np.argsort(similarities)[::-1][:top_n]
    return candidate_indices[order], similarities[order], int(len(candidate_indices))
