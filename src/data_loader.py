"""Data loading and adaptive schema inference utilities."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

MOVIE_TITLE_CANDIDATES = ["title", "movie_title", "movie", "name"]
SONG_TITLE_CANDIDATES = ["title", "song_title", "track", "name"]
ID_CANDIDATES = ["movie_id", "song_id", "item_id", "id", "uid", "movieid", "track_id", "trackid"]
ARTIST_CANDIDATES = ["artist", "singer", "performer", "band"]
TEXT_HINT_COLUMNS = ["tags", "keywords", "description", "summary", "lyrics", "theme", "synopsis"]


def _resolve_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """Resolve a column by candidate names with case-insensitive matching."""
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise ValueError(
            f"Could not resolve required column from candidates {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _find_existing_dir(base_dir: Path, candidates: List[str]) -> Optional[Path]:
    """Return the first existing child directory among candidates."""
    for name in candidates:
        path = base_dir / name
        if path.exists() and path.is_dir():
            return path
    return None


def _find_dir_with_required_files(base_dir: Path, required_files: List[str]) -> Optional[Path]:
    """Return a child directory that contains all required files."""
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if all((child / filename).exists() for filename in required_files):
            return child
    return None


def _join_text(parts: List[object]) -> str:
    """Join non-empty text chunks into one string."""
    cleaned = [str(value).strip() for value in parts if pd.notna(value) and str(value).strip()]
    return " ".join(cleaned)


def _year_to_era(year: Optional[int]) -> str:
    """Map year to coarse era bucket."""
    if year is None or pd.isna(year):
        return "Unknown"
    year_int = int(year)
    if year_int < 1980:
        return "pre-1980"
    if year_int < 1990:
        return "1980s"
    if year_int < 2000:
        return "1990s"
    if year_int < 2010:
        return "2000s"
    if year_int < 2020:
        return "2010s"
    return "2020s"


def _extract_movie_year(title: str) -> Optional[int]:
    """Extract year from MovieLens-style title suffix."""
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    if not match:
        return None
    return int(match.group(1))


def _clean_movie_title(title: str) -> str:
    """Remove trailing '(YYYY)' suffix from movie titles."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def _load_movielens_movies(movielens_dir: Path) -> pd.DataFrame:
    """Build canonical movie dataframe from MovieLens files."""
    movies_path = movielens_dir / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing MovieLens file: {movies_path}")

    movies = pd.read_csv(movies_path)
    if movies.empty:
        raise ValueError("MovieLens movies.csv is empty.")

    movie_id_col = _resolve_column(movies, ["movieId", "movie_id", "id"], required=False) or movies.columns[0]
    title_col = _resolve_column(movies, ["title", "movie_title", "name"], required=True)
    genre_col = _resolve_column(movies, ["genres", "genre"], required=False)

    movies = movies[[movie_id_col, title_col] + ([genre_col] if genre_col else [])].copy()
    movies = movies.rename(columns={movie_id_col: "item_id", title_col: "title"})
    if genre_col:
        movies = movies.rename(columns={genre_col: "genres"})
    else:
        movies["genres"] = "Unknown"

    movies["year"] = movies["title"].map(_extract_movie_year)
    movies["clean_title"] = movies["title"].map(_clean_movie_title)
    movies["era"] = movies["year"].map(_year_to_era)
    movies["genres"] = movies["genres"].fillna("Unknown").astype(str)
    movies["primary_genre"] = movies["genres"].map(
        lambda value: value.split("|")[0] if value and value != "(no genres listed)" else "Unknown"
    )
    movies["genre_text"] = movies["genres"].str.replace("|", " ", regex=False)

    tags_path = movielens_dir / "tags.csv"
    if tags_path.exists():
        tags_df = pd.read_csv(tags_path)
        tags_movie_col = _resolve_column(tags_df, ["movieId", "movie_id", "id"], required=False)
        tag_text_col = _resolve_column(tags_df, ["tag", "tags", "keyword"], required=False)
        if tags_movie_col and tag_text_col:
            tag_map = (
                tags_df[[tags_movie_col, tag_text_col]]
                .dropna()
                .assign(**{tag_text_col: lambda frame: frame[tag_text_col].astype(str).str.lower()})
                .groupby(tags_movie_col)[tag_text_col]
                .apply(lambda values: " ".join(values.head(20).tolist()))
            )
            movies = movies.merge(
                tag_map.rename("ml_tags"),
                left_on="item_id",
                right_index=True,
                how="left",
            )
        else:
            movies["ml_tags"] = ""
    else:
        movies["ml_tags"] = ""

    ratings_path = movielens_dir / "ratings.csv"
    movies["rating_mean"] = 0.0
    movies["rating_count"] = 0.0
    if ratings_path.exists():
        ratings_df = pd.read_csv(ratings_path)
        ratings_movie_col = _resolve_column(ratings_df, ["movieId", "movie_id", "id"], required=False)
        rating_col = _resolve_column(ratings_df, ["rating", "score"], required=False)
        if ratings_movie_col and rating_col:
            stats = ratings_df.groupby(ratings_movie_col)[rating_col].agg(["mean", "count"]).rename(
                columns={"mean": "rating_mean", "count": "rating_count"}
            )
            movies = movies.merge(stats, left_on="item_id", right_index=True, how="left", suffixes=("", "_stats"))
            movies["rating_mean"] = pd.to_numeric(movies["rating_mean_stats"], errors="coerce").fillna(
                pd.to_numeric(movies["rating_mean"], errors="coerce").fillna(0.0)
            )
            movies["rating_count"] = pd.to_numeric(movies["rating_count_stats"], errors="coerce").fillna(
                pd.to_numeric(movies["rating_count"], errors="coerce").fillna(0.0)
            )
            for column in ["rating_mean_stats", "rating_count_stats"]:
                if column in movies.columns:
                    movies = movies.drop(columns=[column])

    movies["popularity"] = np.log1p(pd.to_numeric(movies["rating_count"], errors="coerce").fillna(0.0))
    movies["quality"] = pd.to_numeric(movies["rating_mean"], errors="coerce").fillna(0.0)
    movies["tags"] = movies.apply(
        lambda row: _join_text([row["clean_title"], row["genre_text"], row["ml_tags"]]),
        axis=1,
    )
    keep_columns = ["item_id", "title", "artist", "tags", "primary_genre", "era", "popularity", "quality"]
    movies["artist"] = "N/A"
    movies = movies[keep_columns].rename(columns={"primary_genre": "genre"})
    movies = movies.dropna(subset=["item_id", "title"]).reset_index(drop=True)
    return movies


def _load_lastfm_songs(lastfm_dir: Path, max_songs: int = 15000) -> pd.DataFrame:
    """Build canonical song dataframe from Last.fm listening history."""
    csv_candidates = sorted(lastfm_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV files found in Last.fm directory: {lastfm_dir}")
    lastfm_path = csv_candidates[0]
    listens = pd.read_csv(lastfm_path)
    if listens.empty:
        raise ValueError(f"Last.fm file is empty: {lastfm_path}")

    artist_col = _resolve_column(listens, ["Artist", "artist"], required=True)
    track_col = _resolve_column(listens, ["Track", "title", "song", "name"], required=True)
    album_col = _resolve_column(listens, ["Album", "album"], required=False)
    date_col = _resolve_column(listens, ["Date", "date"], required=False)
    user_col = _resolve_column(listens, ["Username", "user", "userid"], required=False)

    base = listens[
        [artist_col, track_col] + ([album_col] if album_col else []) + ([date_col] if date_col else [])
    ].copy()
    if user_col:
        base[user_col] = listens[user_col]
    base = base.rename(columns={artist_col: "artist", track_col: "track"})
    if album_col:
        base = base.rename(columns={album_col: "album"})
    else:
        base["album"] = ""
    if date_col:
        base = base.rename(columns={date_col: "listen_date"})
    else:
        base["listen_date"] = ""
    if user_col:
        base = base.rename(columns={user_col: "username"})
    else:
        base["username"] = ""

    base["artist"] = base["artist"].fillna("").astype(str).str.strip()
    base["track"] = base["track"].fillna("").astype(str).str.strip()
    base["album"] = base["album"].fillna("").astype(str).str.strip()
    base["username"] = base["username"].fillna("").astype(str).str.strip()
    base = base[(base["artist"] != "") & (base["track"] != "")]

    date_parsed = pd.to_datetime(base["listen_date"], errors="coerce", dayfirst=True)
    base["listen_year"] = date_parsed.dt.year
    grouped = base.groupby(["artist", "track"], as_index=False).agg(
        play_count=("track", "size"),
        album=("album", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
        user_count=("username", lambda values: values[values != ""].nunique()),
        year=("listen_year", "max"),
    )
    grouped = grouped.sort_values("play_count", ascending=False).reset_index(drop=True)
    if max_songs > 0:
        grouped = grouped.head(max_songs).copy()

    grouped["era"] = grouped["year"].map(_year_to_era)
    grouped["title"] = grouped["track"]
    grouped["genre"] = "Unknown"
    grouped["tags"] = grouped.apply(
        lambda row: _join_text([row["track"], row["artist"], row["album"]]),
        axis=1,
    )
    grouped["popularity"] = np.log1p(pd.to_numeric(grouped["play_count"], errors="coerce").fillna(0.0))
    grouped["quality"] = np.log1p(pd.to_numeric(grouped["user_count"], errors="coerce").fillna(0.0))
    grouped["item_id"] = np.arange(1, len(grouped) + 1, dtype=np.int64)

    songs = grouped[["item_id", "title", "artist", "tags", "genre", "era", "popularity", "quality"]]
    songs = songs.dropna(subset=["item_id", "title"]).reset_index(drop=True)
    return songs


def _load_movielens_lastfm_pair(data_path: Path, max_songs: int = 15000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cross-domain movies/songs from MovieLens + Last.fm folders."""
    movielens_dir = _find_existing_dir(
        data_path,
        ["movielens", "movie-lens", "movie lens", "ml-latest-small", "ml_latest_small"],
    )
    if movielens_dir is None:
        movielens_dir = _find_dir_with_required_files(data_path, ["movies.csv", "ratings.csv"])
    lastfm_dir = _find_existing_dir(data_path, ["last fm", "last-fm", "lastfm", "last_fm"])
    if movielens_dir is None or lastfm_dir is None:
        raise FileNotFoundError("Could not find both MovieLens and Last.fm directories in the provided data path.")

    movies_df = _load_movielens_movies(movielens_dir)
    songs_df = _load_lastfm_songs(lastfm_dir, max_songs=max_songs)
    return movies_df, songs_df


def _infer_text_columns(df: pd.DataFrame, shared_columns: List[str]) -> List[str]:
    """Infer rich text columns shared between movies and songs."""
    text_columns = [col for col in shared_columns if col.lower() in {value.lower() for value in TEXT_HINT_COLUMNS}]
    if text_columns:
        return text_columns

    inferred: List[str] = []
    for column in shared_columns:
        series = df[column]
        if not is_string_dtype(series):
            continue
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            continue
        avg_tokens = float(sample.str.split().map(len).mean())
        avg_length = float(sample.str.len().mean())
        if avg_tokens >= 2.0 or avg_length >= 18.0:
            inferred.append(column)
    return inferred


def _infer_feature_types(
    movies_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    movie_id_col: str,
    movie_title_col: str,
    song_id_col: str,
    song_title_col: str,
    artist_col: Optional[str],
) -> Dict[str, List[str]]:
    """Infer shared categorical, numeric, and text feature columns dynamically."""
    excluded_movies = {movie_id_col, movie_title_col}
    excluded_songs = {song_id_col, song_title_col}
    if artist_col is not None:
        excluded_songs.add(artist_col)

    movie_feature_candidates = [column for column in movies_df.columns if column not in excluded_movies]
    song_feature_candidates = [column for column in songs_df.columns if column not in excluded_songs]
    shared_columns = [column for column in movie_feature_candidates if column in song_feature_candidates]

    if not shared_columns:
        movies_df["__title_text"] = movies_df[movie_title_col].astype(str)
        songs_df["__title_text"] = songs_df[song_title_col].astype(str)
        shared_columns = ["__title_text"]

    text_columns = _infer_text_columns(movies_df, shared_columns)
    categorical_columns: List[str] = []
    numeric_columns: List[str] = []

    for column in shared_columns:
        if column in text_columns:
            continue
        movie_series = movies_df[column]
        song_series = songs_df[column]
        if is_numeric_dtype(movie_series) and is_numeric_dtype(song_series):
            numeric_columns.append(column)
            continue
        if is_string_dtype(movie_series) or is_string_dtype(song_series):
            combined = pd.concat([movie_series, song_series], axis=0, ignore_index=True).astype(str)
            unique_count = int(combined.nunique(dropna=True))
            if unique_count <= 250:
                categorical_columns.append(column)
            else:
                text_columns.append(column)

    if not categorical_columns and not numeric_columns and not text_columns:
        movies_df["__title_text"] = movies_df[movie_title_col].astype(str)
        songs_df["__title_text"] = songs_df[song_title_col].astype(str)
        text_columns = ["__title_text"]
        shared_columns = ["__title_text"]

    return {
        "shared_feature_columns": shared_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
    }


def _prepare_movies(
    movies_df: pd.DataFrame,
    movie_id_col: str,
    movie_title_col: str,
    shared_feature_columns: List[str],
) -> pd.DataFrame:
    """Select and normalize movie columns to a canonical schema."""
    keep_columns = [movie_id_col, movie_title_col] + shared_feature_columns
    prepared = movies_df[keep_columns].copy()
    prepared = prepared.rename(columns={movie_id_col: "item_id", movie_title_col: "title"})
    prepared = prepared.dropna(subset=["item_id", "title"]).reset_index(drop=True)
    return prepared


def _prepare_songs(
    songs_df: pd.DataFrame,
    song_id_col: str,
    song_title_col: str,
    artist_col: Optional[str],
    shared_feature_columns: List[str],
) -> pd.DataFrame:
    """Select and normalize song columns to a canonical schema."""
    keep_columns = [song_id_col, song_title_col] + shared_feature_columns
    if artist_col is not None:
        keep_columns.append(artist_col)

    prepared = songs_df[keep_columns].copy()
    rename_map = {song_id_col: "item_id", song_title_col: "title"}
    if artist_col is not None:
        rename_map[artist_col] = "artist"
    prepared = prepared.rename(columns=rename_map)
    if "artist" not in prepared.columns:
        prepared["artist"] = "Unknown Artist"
    prepared = prepared.dropna(subset=["item_id", "title"]).reset_index(drop=True)
    return prepared


def _fill_feature_nulls(
    movies_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    text_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing feature values with robust defaults for each feature type."""
    for column in categorical_columns:
        movies_df[column] = movies_df[column].fillna("Unknown").astype(str)
        songs_df[column] = songs_df[column].fillna("Unknown").astype(str)

    for column in text_columns:
        movies_df[column] = movies_df[column].fillna("").astype(str)
        songs_df[column] = songs_df[column].fillna("").astype(str)

    for column in numeric_columns:
        movie_series = pd.to_numeric(movies_df[column], errors="coerce")
        song_series = pd.to_numeric(songs_df[column], errors="coerce")
        median_value = pd.concat([movie_series, song_series], ignore_index=True).median()
        fill_value = float(median_value) if pd.notna(median_value) else 0.0
        movies_df[column] = movie_series.fillna(fill_value)
        songs_df[column] = song_series.fillna(fill_value)

    songs_df["artist"] = songs_df["artist"].fillna("Unknown Artist").astype(str)
    movies_df["title"] = movies_df["title"].astype(str)
    songs_df["title"] = songs_df["title"].astype(str)
    return movies_df, songs_df


def load_datasets(
    data_dir: str = "data",
    movies_filename: str = "movies.csv",
    songs_filename: str = "songs.csv",
    max_cross_songs: int = 15000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Load datasets and infer a robust shared schema for recommendation training."""
    data_path = Path(data_dir)
    movies_path = data_path / movies_filename
    songs_path = data_path / songs_filename

    source_type = "direct_csv_pair"
    if movies_path.exists() and songs_path.exists():
        movies_raw = pd.read_csv(movies_path)
        songs_raw = pd.read_csv(songs_path)
    else:
        movies_raw, songs_raw = _load_movielens_lastfm_pair(data_path, max_songs=max_cross_songs)
        source_type = "movielens_lastfm_bridge"

    if movies_raw.empty or songs_raw.empty:
        raise ValueError("Input datasets must not be empty.")

    movie_id_col = _resolve_column(movies_raw, ID_CANDIDATES, required=False) or movies_raw.columns[0]
    song_id_col = _resolve_column(songs_raw, ID_CANDIDATES, required=False) or songs_raw.columns[0]
    movie_title_col = _resolve_column(movies_raw, MOVIE_TITLE_CANDIDATES, required=True)
    song_title_col = _resolve_column(songs_raw, SONG_TITLE_CANDIDATES, required=True)
    artist_col = _resolve_column(songs_raw, ARTIST_CANDIDATES, required=False)

    inferred = _infer_feature_types(
        movies_raw,
        songs_raw,
        movie_id_col=movie_id_col,
        movie_title_col=movie_title_col,
        song_id_col=song_id_col,
        song_title_col=song_title_col,
        artist_col=artist_col,
    )
    movies_df = _prepare_movies(
        movies_raw,
        movie_id_col=movie_id_col,
        movie_title_col=movie_title_col,
        shared_feature_columns=inferred["shared_feature_columns"],
    )
    songs_df = _prepare_songs(
        songs_raw,
        song_id_col=song_id_col,
        song_title_col=song_title_col,
        artist_col=artist_col,
        shared_feature_columns=inferred["shared_feature_columns"],
    )
    movies_df, songs_df = _fill_feature_nulls(
        movies_df,
        songs_df,
        categorical_columns=inferred["categorical_columns"],
        numeric_columns=inferred["numeric_columns"],
        text_columns=inferred["text_columns"],
    )

    schema: Dict[str, object] = {
        "movie_id_column": "item_id",
        "song_id_column": "item_id",
        "movie_title_column": "title",
        "song_title_column": "title",
        "artist_column": "artist",
        "shared_feature_columns": inferred["shared_feature_columns"],
        "categorical_columns": inferred["categorical_columns"],
        "numeric_columns": inferred["numeric_columns"],
        "text_columns": inferred["text_columns"],
        "source_type": source_type,
        "max_cross_songs": max_cross_songs,
    }

    if (
        not schema["categorical_columns"]
        and not schema["numeric_columns"]
        and not schema["text_columns"]
    ):
        raise ValueError(
            "No usable shared feature columns found between movies and songs. "
            "Include at least one shared metadata or text column."
        )
    return movies_df.reset_index(drop=True), songs_df.reset_index(drop=True), schema
