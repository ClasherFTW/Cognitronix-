"""Dictionary learning model training and artifact loading."""

import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning


def train_dictionary_model(
    feature_matrix,
    n_components: int = 20,
    alpha: float = 1.0,
    random_state: int = 42,
    use_minibatch: bool = True,
    batch_size: int = 512,
    model_dir: str = "models",
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Train dictionary learning and return learned components, sparse codes, and metrics."""
    start_time = time.perf_counter()
    dense_matrix = (
        feature_matrix.toarray().astype(np.float32) if sparse.issparse(feature_matrix) else np.asarray(feature_matrix)
    )

    if use_minibatch:
        dictionary_model = MiniBatchDictionaryLearning(
            n_components=n_components,
            transform_algorithm="lasso_lars",
            alpha=alpha,
            random_state=random_state,
            batch_size=batch_size,
        )
    else:
        dictionary_model = DictionaryLearning(
            n_components=n_components,
            transform_algorithm="lasso_lars",
            alpha=alpha,
            fit_algorithm="lars",
            random_state=random_state,
        )
    sparse_codes = dictionary_model.fit_transform(dense_matrix)
    dictionary_matrix = dictionary_model.components_

    sparsity_pct = float(np.mean(sparse_codes == 0.0) * 100.0)
    elapsed_seconds = time.perf_counter() - start_time

    if save_artifacts:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(dictionary_model, model_path / "dictionary_model.joblib")
        joblib.dump(dictionary_matrix, model_path / "dictionary.joblib")
        joblib.dump(sparse_codes, model_path / "sparse_codes.joblib")

    print(f"Average sparsity of sparse codes: {sparsity_pct:.2f}% zeros")
    return {
        "model": dictionary_model,
        "dictionary": dictionary_matrix,
        "sparse_codes": sparse_codes,
        "sparsity_pct": sparsity_pct,
        "train_seconds": elapsed_seconds,
    }


def load_trained_artifacts(model_dir: str = "models") -> Dict[str, Any]:
    """Load trained dictionary, sparse codes, preprocessing artifacts, and optional ANN index."""
    model_path = Path(model_dir)
    required_files = {
        "model": model_path / "dictionary_model.joblib",
        "dictionary": model_path / "dictionary.joblib",
        "sparse_codes": model_path / "sparse_codes.joblib",
        "boundaries": model_path / "index_boundaries.joblib",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing model artifacts: {missing_text}. Run `python cli.py train` first."
        )

    artifacts = {
        "model": joblib.load(required_files["model"]),
        "dictionary": joblib.load(required_files["dictionary"]),
        "sparse_codes": joblib.load(required_files["sparse_codes"]),
        "boundaries": joblib.load(required_files["boundaries"]),
    }
    optional_files = {
        "encoder": model_path / "onehot_encoder.joblib",
        "vectorizer": model_path / "tfidf_vectorizer.joblib",
        "scaler": model_path / "numeric_scaler.joblib",
        "feature_config": model_path / "feature_config.joblib",
        "feature_names": model_path / "feature_names.joblib",
        "schema": model_path / "data_schema.joblib",
        "ann_index": model_path / "ann_index.joblib",
    }
    for key, path in optional_files.items():
        if path.exists():
            artifacts[key] = joblib.load(path)
    return artifacts
