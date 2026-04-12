"""CLI entry point for training, recommending, arc playlists, and evaluation."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np

from src.ann_index import build_ann_index
from src.data_loader import load_datasets
from src.dictionary_model import train_dictionary_model
from src.evaluator import evaluate_recommender
from src.preprocessor import fit_feature_space
from src.recommender import format_recommendations_table, generate_arc_playlist, recommend_songs


def _default_paths() -> tuple[Path, Path]:
    """Return default data and model directories relative to this script."""
    base_dir = Path(__file__).resolve().parent
    return base_dir / "data", base_dir / "models"


def _resolve_runtime_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve data and model directories from CLI args with sane defaults."""
    default_data_dir, default_model_dir = _default_paths()
    data_dir = Path(args.data_dir).resolve() if getattr(args, "data_dir", None) else default_data_dir
    model_dir = Path(args.model_dir).resolve() if getattr(args, "model_dir", None) else default_model_dir
    return data_dir, model_dir


def _parse_constraints(items: Iterable[str]) -> Dict[str, List[str]]:
    """Parse repeated `column=value` CLI arguments into a dictionary."""
    constraints: Dict[str, List[str]] = {}
    for item in items:
        if "=" not in item:
            continue
        column, value_text = item.split("=", 1)
        column = column.strip()
        if not column:
            continue
        values = [value.strip() for value in value_text.split("|") if value.strip()]
        if not values:
            continue
        constraints.setdefault(column, []).extend(values)
    return constraints


def _safe_console_text(text: str) -> str:
    """Return text normalized for current console encoding without crashing."""
    encoding = (getattr(sys.stdout, "encoding", None) or "utf-8").strip() or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")


def train_command(args: argparse.Namespace) -> None:
    """Train model artifacts and ANN index."""
    data_dir, model_dir = _resolve_runtime_paths(args)
    model_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    movies_df, songs_df, schema = load_datasets(
        data_dir=str(data_dir),
        movies_filename=args.movies_file,
        songs_filename=args.songs_file,
        max_cross_songs=args.max_cross_songs,
    )
    feature_matrix, boundaries, preprocess_artifacts = fit_feature_space(
        movies_df,
        songs_df,
        schema=schema,
        model_dir=str(model_dir),
        save_artifacts=True,
        max_text_features=args.max_text_features,
        min_text_df=args.min_text_df,
    )
    trained = train_dictionary_model(
        feature_matrix,
        n_components=args.n_components,
        alpha=args.alpha,
        random_state=42,
        use_minibatch=not args.full_batch,
        batch_size=args.batch_size,
        model_dir=str(model_dir),
        save_artifacts=True,
    )

    sparse_codes = np.asarray(trained["sparse_codes"], dtype=np.float32)
    song_codes = sparse_codes[boundaries["song_start"] : boundaries["song_end"]]
    ann_index = build_ann_index(
        song_codes=song_codes,
        n_bits=args.ann_bits,
        random_state=42,
        model_dir=str(model_dir),
        save_artifacts=True,
    )

    total_seconds = time.perf_counter() - start_time
    metadata = {
        "n_components": args.n_components,
        "alpha": args.alpha,
        "num_movies": len(movies_df),
        "num_songs": len(songs_df),
        "total_items": int(feature_matrix.shape[0]),
        "feature_dim": int(feature_matrix.shape[1]),
        "boundaries": boundaries,
        "schema": schema,
        "feature_config": preprocess_artifacts["feature_config"],
        "ann_bits": args.ann_bits,
        "ann_buckets": len(ann_index["buckets"]),
    }
    joblib.dump(metadata, model_dir / "training_metadata.joblib")

    print("\nTraining Summary")
    print(f"Items (movies + songs): {feature_matrix.shape[0]}")
    print(f"Feature dimensions: {feature_matrix.shape[1]}")
    print(f"Sparsity: {trained['sparsity_pct']:.2f}% zeros")
    print(f"ANN buckets: {len(ann_index['buckets'])}")
    print(f"Time taken: {total_seconds:.2f} seconds")
    print(f"Saved artifacts to: {model_dir}")


def recommend_command(args: argparse.Namespace) -> None:
    """Generate top-N song recommendations with counterfactual controls."""
    data_dir, model_dir = _resolve_runtime_paths(args)
    try:
        result = recommend_songs(
            movie_title=args.movie,
            top_n=args.top,
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            movies_filename=args.movies_file,
            songs_filename=args.songs_file,
            max_cross_songs=args.max_cross_songs,
            prefer=_parse_constraints(args.prefer),
            avoid=_parse_constraints(args.avoid),
            boost_terms=args.boost_term,
            counterfactual_strength=args.cf_strength,
            explain=args.explain,
            use_ann=not args.no_ann,
            ann_search_k=args.ann_search_k,
            ann_hamming_radius=args.ann_radius,
            diversify=not args.no_diversify,
        )
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    if result["matched_movie"] is None:
        print("No close movie match found.")
        suggestions = result["suggestions"][:5]
        if suggestions:
            print("Did you mean one of these?")
            for title in suggestions:
                print(f"- {title}")
        return

    matched = result["matched_movie"]
    print(_safe_console_text(f"Matched movie: {matched['title']} (score={matched['score']:.1f})"))
    if "candidate_pool_size" in result:
        print(f"Candidate pool size: {result['candidate_pool_size']}\n")
    else:
        print()
    print(_safe_console_text(format_recommendations_table(result["results"])))


def arc_command(args: argparse.Namespace) -> None:
    """Generate staged cinematic arc playlist."""
    data_dir, model_dir = _resolve_runtime_paths(args)
    try:
        result = generate_arc_playlist(
            movie_title=args.movie,
            stages=args.stages,
            per_stage=args.per_stage,
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            movies_filename=args.movies_file,
            songs_filename=args.songs_file,
            max_cross_songs=args.max_cross_songs,
            use_ann=not args.no_ann,
            ann_search_k=args.ann_search_k,
        )
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    if result["matched_movie"] is None:
        print("No close movie match found.")
        suggestions = result["suggestions"][:5]
        if suggestions:
            print("Did you mean one of these?")
            for title in suggestions:
                print(f"- {title}")
        return

    matched = result["matched_movie"]
    print(_safe_console_text(f"Matched movie: {matched['title']} (score={matched['score']:.1f})\n"))
    print(_safe_console_text(format_recommendations_table(result["results"])))


def evaluate_command(args: argparse.Namespace) -> None:
    """Run hold-out evaluation and print accuracy + hackathon metrics."""
    data_dir, model_dir = _resolve_runtime_paths(args)
    summary_df, test_count = evaluate_recommender(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        movies_filename=args.movies_file,
        songs_filename=args.songs_file,
        max_cross_songs=args.max_cross_songs,
        n_components=args.n_components,
        alpha=args.alpha,
        k_values=(5, 10),
        test_size=0.2,
        random_state=42,
        max_text_features=args.max_text_features,
        min_text_df=args.min_text_df,
    )

    printable = summary_df.copy()
    metric_columns = [column for column in printable.columns if column != "K"]
    for column in metric_columns:
        printable[column] = printable[column].map(lambda value: f"{value:.4f}")

    print(f"Evaluation completed on {test_count} held-out movies.\n")
    print(printable.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Adaptive dictionary-learning movie-to-song recommender."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_data_arguments = [
        (["--data_dir"], {"type": str, "default": None, "help": "Dataset root directory (default: ./data)."}),
        (["--model_dir"], {"type": str, "default": None, "help": "Model artifact directory (default: ./models)."}),
        (["--movies_file"], {"type": str, "default": "movies.csv", "help": "Movies CSV filename inside data_dir."}),
        (["--songs_file"], {"type": str, "default": "songs.csv", "help": "Songs CSV filename inside data_dir."}),
        (
            ["--max_cross_songs"],
            {"type": int, "default": 15000, "help": "When auto-bridging MovieLens+Last.fm, keep top-N Last.fm tracks."},
        ),
    ]

    train_parser = subparsers.add_parser("train", help="Train and save model artifacts.")
    train_parser.add_argument("--n_components", type=int, default=32, help="Dictionary atoms.")
    train_parser.add_argument("--alpha", type=float, default=0.25, help="Sparsity regularization.")
    train_parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size.")
    train_parser.add_argument("--full_batch", action="store_true", help="Use classic DictionaryLearning.")
    train_parser.add_argument("--max_text_features", type=int, default=3000, help="Max TF-IDF vocabulary size.")
    train_parser.add_argument("--min_text_df", type=int, default=1, help="Min document frequency for text tokens.")
    train_parser.add_argument("--ann_bits", type=int, default=24, help="Number of LSH bits for ANN index.")
    for flags, kwargs in common_data_arguments:
        train_parser.add_argument(*flags, **kwargs)
    train_parser.set_defaults(func=train_command)

    recommend_parser = subparsers.add_parser("recommend", help="Recommend songs for a movie.")
    recommend_parser.add_argument("--movie", required=True, help="Movie title query.")
    recommend_parser.add_argument("--top", type=int, default=5, help="Number of songs to return.")
    recommend_parser.add_argument(
        "--prefer",
        action="append",
        default=[],
        help="Counterfactual preference, format: column=value or column=v1|v2.",
    )
    recommend_parser.add_argument(
        "--avoid",
        action="append",
        default=[],
        help="Counterfactual avoid rule, format: column=value or column=v1|v2.",
    )
    recommend_parser.add_argument(
        "--boost_term",
        action="append",
        default=[],
        help="Boost songs containing these text terms.",
    )
    recommend_parser.add_argument("--cf_strength", type=float, default=0.35, help="Counterfactual shift strength.")
    recommend_parser.add_argument("--explain", action="store_true", help="Show latent atom explanations.")
    recommend_parser.add_argument("--no_ann", action="store_true", help="Disable ANN and use exact retrieval.")
    recommend_parser.add_argument("--ann_search_k", type=int, default=256, help="ANN candidate pool size.")
    recommend_parser.add_argument("--ann_radius", type=int, default=2, help="ANN Hamming search radius.")
    recommend_parser.add_argument("--no_diversify", action="store_true", help="Disable diversity reranking.")
    for flags, kwargs in common_data_arguments:
        recommend_parser.add_argument(*flags, **kwargs)
    recommend_parser.set_defaults(func=recommend_command)

    arc_parser = subparsers.add_parser("arc", help="Generate a cinematic multi-stage playlist.")
    arc_parser.add_argument("--movie", required=True, help="Movie title query.")
    arc_parser.add_argument("--stages", type=int, default=3, help="Number of narrative stages.")
    arc_parser.add_argument("--per_stage", type=int, default=3, help="Songs per stage.")
    arc_parser.add_argument("--no_ann", action="store_true", help="Disable ANN and use exact retrieval.")
    arc_parser.add_argument("--ann_search_k", type=int, default=512, help="ANN candidate pool size.")
    for flags, kwargs in common_data_arguments:
        arc_parser.add_argument(*flags, **kwargs)
    arc_parser.set_defaults(func=arc_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate with hold-out movies.")
    evaluate_parser.add_argument("--n_components", type=int, default=32, help="Dictionary atoms.")
    evaluate_parser.add_argument("--alpha", type=float, default=0.25, help="Sparsity regularization.")
    evaluate_parser.add_argument("--max_text_features", type=int, default=3000, help="Max TF-IDF vocabulary size.")
    evaluate_parser.add_argument("--min_text_df", type=int, default=1, help="Min document frequency for text tokens.")
    for flags, kwargs in common_data_arguments:
        evaluate_parser.add_argument(*flags, **kwargs)
    evaluate_parser.set_defaults(func=evaluate_command)
    return parser


def main() -> None:
    """CLI main function."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
