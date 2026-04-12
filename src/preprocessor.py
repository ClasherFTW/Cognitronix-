"""Adaptive feature preprocessing module for movies and songs."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with different scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _ensure_csr(matrix) -> sparse.csr_matrix:
    """Convert a matrix to CSR sparse format."""
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def _empty_sparse(num_rows: int) -> sparse.csr_matrix:
    """Create an empty sparse matrix with `num_rows` rows."""
    return sparse.csr_matrix((num_rows, 0), dtype=np.float32)


def _combine_text_columns(df: pd.DataFrame, text_columns: List[str]) -> pd.Series:
    """Combine text columns into one corpus string with column-aware prefixes."""
    if not text_columns:
        return pd.Series([""] * len(df), index=df.index, dtype=str)

    combined = pd.Series([""] * len(df), index=df.index, dtype=str)
    for column in text_columns:
        chunk = df[column].fillna("").astype(str).map(str.strip)
        combined = combined + " " + chunk.map(lambda value: f"{column}:{value}" if value else "")
    return combined.str.strip()


def _onehot_feature_names(encoder: OneHotEncoder, categorical_columns: List[str]) -> List[str]:
    """Extract stable feature names from fitted one-hot encoder."""
    if not categorical_columns:
        return []
    try:
        names = encoder.get_feature_names_out(categorical_columns)
    except AttributeError:
        names = encoder.get_feature_names(categorical_columns)
    return [f"cat:{name}" for name in names.tolist()]


def fit_feature_space(
    movies_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    schema: Dict[str, Any],
    model_dir: str = "models",
    save_artifacts: bool = True,
    max_text_features: int = 3000,
    min_text_df: int = 1,
) -> Tuple[sparse.csr_matrix, Dict[str, int], Dict[str, Any]]:
    """Fit adaptive feature transformers over combined movies and songs."""
    combined_df = pd.concat([movies_df, songs_df], ignore_index=True, sort=False)
    categorical_columns = list(schema.get("categorical_columns", []))
    numeric_columns = list(schema.get("numeric_columns", []))
    text_columns = list(schema.get("text_columns", []))

    encoder = _build_one_hot_encoder() if categorical_columns else None
    vectorizer: Any = TfidfVectorizer(
        max_features=max_text_features,
        min_df=min_text_df,
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    scaler = MaxAbsScaler() if numeric_columns else None

    matrices: List[sparse.csr_matrix] = []
    feature_names: List[str] = []

    if categorical_columns:
        categorical_matrix = _ensure_csr(encoder.fit_transform(combined_df[categorical_columns])).astype(np.float32)
        matrices.append(categorical_matrix)
        feature_names.extend(_onehot_feature_names(encoder, categorical_columns))
    else:
        categorical_matrix = _empty_sparse(len(combined_df))

    text_corpus = _combine_text_columns(combined_df, text_columns)
    if text_corpus.str.len().sum() > 0:
        text_matrix = _ensure_csr(vectorizer.fit_transform(text_corpus)).astype(np.float32)
        matrices.append(text_matrix)
        feature_names.extend([f"txt:{token}" for token in vectorizer.get_feature_names_out().tolist()])
    else:
        text_matrix = _empty_sparse(len(combined_df))
        vectorizer = None

    if numeric_columns:
        numeric_values = combined_df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        )
        scaled_numeric = scaler.fit_transform(numeric_values)
        numeric_matrix = _ensure_csr(scaled_numeric).astype(np.float32)
        matrices.append(numeric_matrix)
        feature_names.extend([f"num:{column}" for column in numeric_columns])
    else:
        numeric_matrix = _empty_sparse(len(combined_df))

    if not matrices:
        fallback_vectorizer = TfidfVectorizer(dtype=np.float32)
        fallback_matrix = fallback_vectorizer.fit_transform(combined_df["title"].astype(str))
        matrices.append(_ensure_csr(fallback_matrix))
        vectorizer = fallback_vectorizer
        text_columns = ["title"]
        feature_names.extend([f"txt:{token}" for token in vectorizer.get_feature_names_out().tolist()])

    feature_matrix = sparse.hstack(matrices, format="csr").astype(np.float32)
    boundaries = {
        "movie_start": 0,
        "movie_end": len(movies_df),
        "song_start": len(movies_df),
        "song_end": len(combined_df),
    }

    preprocess_artifacts: Dict[str, Any] = {
        "encoder": encoder,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "feature_config": {
            "categorical_columns": categorical_columns,
            "numeric_columns": numeric_columns,
            "text_columns": text_columns,
        },
        "feature_names": feature_names,
        "boundaries": boundaries,
        "schema": schema,
    }

    if save_artifacts:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, model_path / "onehot_encoder.joblib")
        joblib.dump(vectorizer, model_path / "tfidf_vectorizer.joblib")
        joblib.dump(scaler, model_path / "numeric_scaler.joblib")
        joblib.dump(preprocess_artifacts["feature_config"], model_path / "feature_config.joblib")
        joblib.dump(feature_names, model_path / "feature_names.joblib")
        joblib.dump(boundaries, model_path / "index_boundaries.joblib")
        joblib.dump(schema, model_path / "data_schema.joblib")

    return feature_matrix, boundaries, preprocess_artifacts


def transform_with_feature_space(items_df: pd.DataFrame, preprocess_artifacts: Dict[str, Any]) -> sparse.csr_matrix:
    """Transform new items into the same adaptive feature space as training."""
    feature_config = preprocess_artifacts["feature_config"]
    categorical_columns = list(feature_config.get("categorical_columns", []))
    numeric_columns = list(feature_config.get("numeric_columns", []))
    text_columns = list(feature_config.get("text_columns", []))

    matrices: List[sparse.csr_matrix] = []
    encoder = preprocess_artifacts.get("encoder")
    vectorizer = preprocess_artifacts.get("vectorizer")
    scaler = preprocess_artifacts.get("scaler")

    if categorical_columns and encoder is not None:
        categorical_matrix = _ensure_csr(encoder.transform(items_df[categorical_columns])).astype(np.float32)
        matrices.append(categorical_matrix)

    text_corpus = _combine_text_columns(items_df, text_columns)
    if vectorizer is not None:
        text_matrix = _ensure_csr(vectorizer.transform(text_corpus)).astype(np.float32)
        matrices.append(text_matrix)

    if numeric_columns and scaler is not None:
        numeric_values = items_df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        )
        scaled_numeric = scaler.transform(numeric_values)
        numeric_matrix = _ensure_csr(scaled_numeric).astype(np.float32)
        matrices.append(numeric_matrix)

    if not matrices:
        return _empty_sparse(len(items_df))
    return sparse.hstack(matrices, format="csr").astype(np.float32)
