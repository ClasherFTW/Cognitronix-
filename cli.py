"""CLI entry point for training, recommending, and evaluating the system."""

import argparse
import time
from pathlib import Path

import joblib

from src.data_loader import load_datasets
from src.dictionary_model import train_dictionary_model
from src.evaluator import evaluate_recommender
from src.preprocessor import fit_feature_space
from src.recommender import format_recommendations_table, recommend_songs


def _default_paths() -> tuple[Path, Path]:
    """Return default data and model directories relative to this script."""
    base_dir = Path(__file__).resolve().parent
    return base_dir / "data", base_dir / "models"


def train_command(args: argparse.Namespace) -> None:
    """Train the full recommendation model and persist artifacts."""
    data_dir, model_dir = _default_paths()
    model_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    movies_df, songs_df = load_datasets(str(data_dir))
    feature_matrix, boundaries, _, _ = fit_feature_space(
        movies_df, songs_df, model_dir=str(model_dir), save_artifacts=True
    )
    trained = train_dictionary_model(
        feature_matrix,
        n_components=args.n_components,
        alpha=args.alpha,
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
    }
    joblib.dump(metadata, model_dir / "training_metadata.joblib")

    print("\nTraining Summary")
    print(f"Items (movies + songs): {feature_matrix.shape[0]}")
    print(f"Feature dimensions: {feature_matrix.shape[1]}")
    print(f"Sparsity: {trained['sparsity_pct']:.2f}% zeros")
    print(f"Time taken: {total_seconds:.2f} seconds")
    print(f"Saved artifacts to: {model_dir}")


def recommend_command(args: argparse.Namespace) -> None:
    """Generate and print top-N song recommendations for a movie title."""
    data_dir, model_dir = _default_paths()
    try:
        result = recommend_songs(
            movie_title=args.movie,
            top_n=args.top,
            data_dir=str(data_dir),
            model_dir=str(model_dir),
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
    print(f"Matched movie: {matched['title']} (score={matched['score']:.1f})\n")
    print(format_recommendations_table(result["results"]))


def evaluate_command(args: argparse.Namespace) -> None:
    """Run hold-out evaluation and print a summary table for K=5 and K=10."""
    data_dir, model_dir = _default_paths()
    summary_df, test_count = evaluate_recommender(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        n_components=args.n_components,
        alpha=args.alpha,
        k_values=(5, 10),
        test_size=0.2,
        random_state=42,
    )

    printable = summary_df.copy()
    for column in ["Precision@K", "Recall@K", "F1@K", "NDCG@K"]:
        printable[column] = printable[column].map(lambda value: f"{value:.4f}")

    print(f"Evaluation completed on {test_count} held-out movies.\n")
    print(printable.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Dictionary learning + sparse representation recommender system."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save model artifacts.")
    train_parser.add_argument("--n_components", type=int, default=20, help="Dictionary atoms.")
    train_parser.add_argument("--alpha", type=float, default=1.0, help="Sparsity regularization.")
    train_parser.set_defaults(func=train_command)

    recommend_parser = subparsers.add_parser("recommend", help="Recommend songs for a movie.")
    recommend_parser.add_argument("--movie", required=True, help="Movie title query.")
    recommend_parser.add_argument("--top", type=int, default=5, help="Number of songs to return.")
    recommend_parser.set_defaults(func=recommend_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate with hold-out movies.")
    evaluate_parser.add_argument("--n_components", type=int, default=20, help="Dictionary atoms.")
    evaluate_parser.add_argument("--alpha", type=float, default=1.0, help="Sparsity regularization.")
    evaluate_parser.set_defaults(func=evaluate_command)
    return parser


def main() -> None:
    """CLI main function."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
