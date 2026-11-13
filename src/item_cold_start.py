
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
POSITIVE_THRESHOLD = 4.0          # rating >= 4.0 counts as "liked" / "relevant"

RATINGS_COLUMNS = ["userID", "movieID", "rating", "date_day", "date_month",
                   "date_year", "date_hour", "date_minute", "date_second"]
MOVIE_COLUMNS = ["movieID", "title", "genres", "tags", "actors", "directors",
                 "countries", "content"]
METADATA_FIELDS = ["genres", "tags", "actors", "directors", "countries"]

CATEGORY_COLD = "cold_start"          # 0 training ratings
CATEGORY_LIMITED = "limited_rating"   # 1-5 training ratings
CATEGORY_WARM = "warm"                # > 5 training ratings
CATEGORIES = [CATEGORY_COLD, CATEGORY_LIMITED, CATEGORY_WARM]

RECOMMENDATION_COLUMNS = ["movieID", "title", "recommendation_score"]
SIMILARITY_COLUMNS = ["movieID", "title", "similarity_score", "recommendation_score"]

PROCESSED_FILES = {
    "ratings": "ratings_clean.csv",
    "movies": "movie_features.csv",
    "train": "train_ratings.csv",
    "test": "test_ratings.csv",
}


# --------------------------------------------------------------------------
# 0. Paths, loading and validation
# --------------------------------------------------------------------------
def get_project_paths(project_root: Optional[Path] = None,
                      create_output_dirs: bool = True) -> Dict[str, Path]:
    
    if project_root is None:
        start = Path.cwd().resolve()
        for candidate in [start, *start.parents]:
            if (candidate / "dataset" / "processed").is_dir():
                project_root = candidate
                break
        else:
            raise FileNotFoundError(
                "Could not find 'dataset/processed' in the current folder or any "
                "parent folder. Run the notebook inside your project repository, "
                "or pass project_root=Path('/path/to/your/project')."
            )
    root = Path(project_root).resolve()
    paths = {
        "root": root,
        "data_dir": root / "dataset" / "processed",
        "results_dir": root / "outputs" / "results",
        "figures_dir": root / "outputs" / "figures",
    }
    if create_output_dirs:
        paths["results_dir"].mkdir(parents=True, exist_ok=True)
        paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def validate_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
   
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required column(s): {missing}. "
            f"Columns found: {list(df.columns)}"
        )


def check_movieid_dtypes(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
   
    rows = [{"dataset": n, "movieID_dtype": str(f["movieID"].dtype)} for n, f in frames.items()]
    return pd.DataFrame(rows)


def load_processed_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    
    data_dir = Path(data_dir)
    frames: Dict[str, pd.DataFrame] = {}
    for key, filename in PROCESSED_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Place the shared processed datasets in {data_dir}: "
                f"{list(PROCESSED_FILES.values())}"
            )
        frames[key] = pd.read_csv(path)

    for key in ("ratings", "train", "test"):
        validate_columns(frames[key], RATINGS_COLUMNS, PROCESSED_FILES[key])
    validate_columns(frames["movies"], MOVIE_COLUMNS, PROCESSED_FILES["movies"])

    dtypes = check_movieid_dtypes({PROCESSED_FILES[k]: v for k, v in frames.items()})
    if dtypes["movieID_dtype"].nunique() != 1:
        raise ValueError("movieID dtype differs between files:\n" + dtypes.to_string(index=False))
    return frames


# --------------------------------------------------------------------------
# 1. Metadata preparation helpers
# --------------------------------------------------------------------------
def _clean_text(series: pd.Series) -> pd.Series:
    
    return (series.fillna("").astype(str)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def _is_available(series: pd.Series) -> pd.Series:
   
    return series.notna() & (series.astype(str).str.strip() != "")


def prepare_movie_features(movie_features: pd.DataFrame) -> pd.DataFrame:
    
    validate_columns(movie_features, ["movieID", "title"], "movie_features")
    if movie_features["movieID"].duplicated().any():
        raise ValueError("movie_features contains duplicated movieID values.")

    mf = movie_features.copy().reset_index(drop=True)
    for col in METADATA_FIELDS + ["content"]:
        mf[col] = _clean_text(mf[col]) if col in mf.columns else ""
    mf["title"] = mf["title"].fillna("").astype(str)

    combined = mf[METADATA_FIELDS].agg(" ".join, axis=1)
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()
    empty_content = mf["content"] == ""
    mf.loc[empty_content, "content"] = combined[empty_content]
    return mf


def build_movie_index(movie_features: pd.DataFrame) -> Dict[int, int]:
   
    return {int(m): i for i, m in enumerate(movie_features["movieID"].to_numpy())}


# --------------------------------------------------------------------------
# 2. Cold-start simulation
# --------------------------------------------------------------------------
def simulate_item_cold_start(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    movie_features: pd.DataFrame,
    n_cold: int = 300,
    n_limited: int = 300,
    n_warm: int = 300,
    limited_max_train: int = 5,
    warm_min_train: int = 20,
    min_total_interactions: int = 20,
    min_heldout_interactions: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict:
    
    validate_columns(train_ratings, ["userID", "movieID", "rating"], "train_ratings")
    validate_columns(test_ratings, ["userID", "movieID", "rating"], "test_ratings")
    rng = np.random.default_rng(random_state)

    train = train_ratings.reset_index(drop=True).copy()
    test = test_ratings.reset_index(drop=True).copy()
    catalogue = set(movie_features["movieID"].tolist())

    # ---- per-movie interaction counts --------------------------------------
    stats = pd.DataFrame({
        "n_train": train.groupby("movieID").size(),
        "n_test": test.groupby("movieID").size(),
    }).fillna(0).astype(int).sort_index()
    stats = stats[stats.index.isin(catalogue)]
    stats["n_total"] = stats["n_train"] + stats["n_test"]
    stats.index.name = "movieID"

    def draw(pool_mask: pd.Series, size: int, taken: set, label: str) -> np.ndarray:
        pool = stats.index[pool_mask & ~stats.index.isin(list(taken))].to_numpy()
        if len(pool) < size:
            warnings.warn(f"Only {len(pool)} eligible movies for '{label}' (requested {size}).")
            size = len(pool)
        return np.sort(rng.choice(pool, size=size, replace=False))

    # ---- select the three disjoint groups ----------------------------------
    cold = draw(stats["n_total"] >= min_total_interactions, n_cold, set(), CATEGORY_COLD)
    taken = set(cold.tolist())
    limited = draw(stats["n_total"] >= min_total_interactions, n_limited, taken, CATEGORY_LIMITED)
    taken |= set(limited.tolist())
    warm = draw((stats["n_train"] >= warm_min_train) & (stats["n_test"] >= min_heldout_interactions),
                n_warm, taken, CATEGORY_WARM)
    groups = {CATEGORY_COLD: cold, CATEGORY_LIMITED: limited, CATEGORY_WARM: warm}
    category_of = {int(m): cat for cat, ids in groups.items() for m in ids}

    # ---- cold: remove ALL training ratings ----------------------------------
    cold_rows = train[train["movieID"].isin(cold)]

    # ---- limited: keep a random <= limited_max_train per movie ---------------
    lim_rows = train[train["movieID"].isin(limited)]
    random_key = pd.Series(rng.random(len(lim_rows)), index=lim_rows.index)
    order_in_movie = random_key.groupby(lim_rows["movieID"]).rank(method="first")
    moved_limited = lim_rows[order_in_movie > limited_max_train]

    moved = pd.concat([cold_rows.assign(source="removed_from_train"),
                       moved_limited.assign(source="removed_from_train")])
    sim_train = train.drop(index=moved.index).reset_index(drop=True)

    # ---- held-out interactions ----------------------------------------------
    selected = np.concatenate([cold, limited, warm])
    orig_test = test[test["movieID"].isin(selected)].assign(source="original_test")
    heldout = pd.concat([moved, orig_test], ignore_index=True)
    heldout["category"] = heldout["movieID"].map(category_of)

    # ---- movie-level summary -------------------------------------------------
    sim_counts = sim_train.groupby("movieID").size()
    held_counts = heldout.groupby("movieID").size()
    summary = stats.loc[selected].copy()
    summary["category"] = summary.index.map(category_of)
    summary["n_train_simulated"] = sim_counts.reindex(summary.index).fillna(0).astype(int)
    summary["n_heldout"] = held_counts.reindex(summary.index).fillna(0).astype(int)
    summary = summary.reset_index().rename(columns={"n_train": "n_train_original",
                                                    "n_test": "n_test_original"})

    sim = {
        "train": sim_train, "heldout": heldout, "groups": groups, "summary": summary,
        "params": dict(n_cold=n_cold, n_limited=n_limited, n_warm=n_warm,
                       limited_max_train=limited_max_train, warm_min_train=warm_min_train,
                       min_total_interactions=min_total_interactions,
                       min_heldout_interactions=min_heldout_interactions,
                       random_state=random_state),
    }
    problems = [k for k, ok in validate_simulation(sim).items() if not ok]
    if problems:
        raise RuntimeError(f"Cold-start simulation failed these integrity checks: {problems}")
    return sim


def validate_simulation(sim: dict) -> Dict[str, bool]:
    
    p = sim["params"]
    s = sim["summary"]
    train_pairs = set(zip(sim["train"]["userID"], sim["train"]["movieID"]))
    held_pairs = set(zip(sim["heldout"]["userID"], sim["heldout"]["movieID"]))
    cold, lim, warm = (s[s["category"] == c] for c in CATEGORIES)
    all_ids = [set(sim["groups"][c].tolist()) for c in CATEGORIES]
    return {
        "cold movies have 0 training ratings": bool((cold["n_train_simulated"] == 0).all()),
        "limited movies have 1..limited_max_train ratings": bool(
            lim["n_train_simulated"].between(1 if (lim["n_train_original"] > 0).all() else 0,
                                             p["limited_max_train"]).all()),
        "warm movies have > 5 training ratings": bool((warm["n_train_simulated"] > 5).all()),
        "no user-movie pair is in both train and held-out": train_pairs.isdisjoint(held_pairs),
        "groups are disjoint": not (all_ids[0] & all_ids[1] or all_ids[0] & all_ids[2]
                                    or all_ids[1] & all_ids[2]),
        "every selected movie has held-out interactions": bool((s["n_heldout"] > 0).all()),
    }


# --------------------------------------------------------------------------
# 3. Metadata analysis
# --------------------------------------------------------------------------
def analyze_metadata_coverage(movie_features: pd.DataFrame,
                              scopes: Optional[Dict[str, Iterable[int]]] = None) -> pd.DataFrame:
    
    validate_columns(movie_features, ["movieID"] + METADATA_FIELDS, "movie_features")
    all_scopes: Dict[str, Optional[Iterable[int]]] = {"All movies": None}
    all_scopes.update(scopes or {})

    rows = []
    for label, ids in all_scopes.items():
        subset = movie_features if ids is None else movie_features[movie_features["movieID"].isin(list(ids))]
        total = len(subset)
        flags = pd.DataFrame({f: _is_available(subset[f]) for f in METADATA_FIELDS})
        for f in METADATA_FIELDS:
            n = int(flags[f].sum())
            rows.append((label, total, f, n, total - n, 100.0 * n / total if total else np.nan))
        pct = [r[5] for r in rows[-len(METADATA_FIELDS):]]
        rows.append((label, total, "overall_coverage", np.nan, np.nan, float(np.mean(pct))))
        n_all = int(flags.all(axis=1).sum())
        rows.append((label, total, "all_five_fields", n_all, total - n_all,
                     100.0 * n_all / total if total else np.nan))
    return pd.DataFrame(rows, columns=["scope", "total_movies", "metadata_field",
                                       "available_count", "missing_count", "availability_pct"])


# --------------------------------------------------------------------------
# 4. TF-IDF
# --------------------------------------------------------------------------
def build_tfidf_model(movie_features: pd.DataFrame, feature_column: str,
                      **tfidf_kwargs) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
   
    if feature_column not in movie_features.columns:
        raise KeyError(f"Column '{feature_column}' not found in movie_features.")
    mf = prepare_movie_features(movie_features) if feature_column in METADATA_FIELDS + ["content"] \
        else movie_features
    text = mf[feature_column].fillna("").astype(str)

    if feature_column == "genres":
        params = dict(token_pattern=r"\S+", lowercase=True, sublinear_tf=True)
    else:
        params = dict(stop_words="english", min_df=2, lowercase=True, sublinear_tf=True)
    params.update(tfidf_kwargs)

    vectorizer = TfidfVectorizer(**params)
    try:
        matrix = vectorizer.fit_transform(text)
    except ValueError as err:
        raise ValueError(f"TF-IDF failed for column '{feature_column}': {err}") from err
    return vectorizer, matrix.tocsr()


_MODEL_CACHE: Dict[tuple, tuple] = {}


def _get_cached_model(movie_features: pd.DataFrame, feature_column: str):
   
    cols = ["movieID"] + [c for c in METADATA_FIELDS + ["content"] if c in movie_features.columns]
    key = (feature_column, len(movie_features),
           int(pd.util.hash_pandas_object(movie_features[cols], index=False).sum()))
    if key not in _MODEL_CACHE:
        if len(_MODEL_CACHE) >= 8:
            _MODEL_CACHE.clear()
        mf = prepare_movie_features(movie_features)
        _, matrix = build_tfidf_model(mf, feature_column)
        _MODEL_CACHE[key] = (mf, matrix, build_movie_index(mf))
    return _MODEL_CACHE[key]


# --------------------------------------------------------------------------
# 5. Cosine similarity and item-to-item recommendation
# --------------------------------------------------------------------------
def calculate_similarity(query_vectors, item_matrix) -> np.ndarray:
   
    return cosine_similarity(query_vectors, item_matrix, dense_output=True)


def _empty(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64" if "score" in c else "object") for c in columns})


def recommend_similar_movies(movie_id: int, movie_features: pd.DataFrame, top_n: int = 10,
                             feature_column: str = "content",
                             tfidf_matrix: Optional[sparse.spmatrix] = None,
                             movie_id_to_index: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    
    if not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    if tfidf_matrix is None:
        mf, tfidf_matrix, movie_id_to_index = _get_cached_model(movie_features, feature_column)
    else:
        mf = movie_features.reset_index(drop=True)
        movie_id_to_index = movie_id_to_index or build_movie_index(mf)

    try:
        row = movie_id_to_index[int(movie_id)]
    except (KeyError, TypeError, ValueError):
        warnings.warn(f"movieID {movie_id!r} was not found in movie_features.")
        return _empty(SIMILARITY_COLUMNS)

    if tfidf_matrix[row].nnz == 0:
        warnings.warn(f"movieID {movie_id!r} has no usable '{feature_column}' metadata; "
                      "no similarity can be computed.")
        return _empty(SIMILARITY_COLUMNS)

    scores = calculate_similarity(tfidf_matrix[row], tfidf_matrix)[0]
    scores[row] = -np.inf                                   # exclude the movie itself
    order = np.argsort(-scores, kind="stable")[:top_n]      # ties -> catalogue order
    order = order[np.isfinite(scores[order])]
    top = mf.iloc[order]
    return pd.DataFrame({
        "movieID": top["movieID"].to_numpy(),
        "title": top["title"].to_numpy(),
        "similarity_score": scores[order],
        "recommendation_score": scores[order],
    })


# --------------------------------------------------------------------------
# 6. User preference profile and user-based cold-start recommendation
# --------------------------------------------------------------------------
def build_user_profile(user_id: int, train_ratings: pd.DataFrame, movie_features: pd.DataFrame,
                       tfidf_matrix: sparse.spmatrix,
                       movie_id_to_index: Optional[Dict[int, int]] = None,
                       positive_threshold: float = POSITIVE_THRESHOLD) -> Optional[sparse.csr_matrix]:
    
    if movie_id_to_index is None:
        movie_id_to_index = build_movie_index(movie_features)

    rows = train_ratings.loc[train_ratings["userID"] == user_id, ["movieID", "rating"]]
    rows = rows.dropna(subset=["rating"])
    positive = rows[rows["rating"] >= positive_threshold]
    if positive.empty:
        return None

    idx = positive["movieID"].map(movie_id_to_index)
    known = idx.notna().to_numpy()                      # skip movies without a matrix row
    if not known.any():
        return None
    row_idx = idx[known].to_numpy(dtype=int)
    weights = positive["rating"].to_numpy(dtype=float)[known]

    profile = sparse.csr_matrix(weights) @ tfidf_matrix[row_idx]   # (1,n) @ (n,F) -> (1,F)
    profile = sparse.csr_matrix(profile / weights.sum())
    return profile if profile.nnz > 0 else None


def recommend_cold_start_movies(user_id: int, train_ratings: pd.DataFrame,
                                movie_features: pd.DataFrame,
                                cold_start_movie_ids: Iterable[int],
                                tfidf_matrix: sparse.spmatrix,
                                movie_id_to_index: Dict[int, int],
                                top_n: int = 10) -> pd.DataFrame:
    
    profile = build_user_profile(user_id, train_ratings, movie_features,
                                 tfidf_matrix, movie_id_to_index)
    if profile is None:
        return _empty(RECOMMENDATION_COLUMNS)

    already_rated = set(train_ratings.loc[train_ratings["userID"] == user_id, "movieID"].tolist())
    candidates = [m for m in dict.fromkeys(cold_start_movie_ids)
                  if m in movie_id_to_index and m not in already_rated]
    if not candidates:
        return _empty(RECOMMENDATION_COLUMNS)

    cand_rows = np.fromiter((movie_id_to_index[m] for m in candidates), dtype=int, count=len(candidates))
    scores = calculate_similarity(profile, tfidf_matrix[cand_rows])[0]
    order = np.argsort(-scores, kind="stable")[:top_n]          # ties -> candidate order
    top = movie_features.iloc[cand_rows[order]]
    return pd.DataFrame({
        "movieID": top["movieID"].to_numpy(),
        "title": top["title"].to_numpy(),
        "recommendation_score": scores[order],
    })


def make_random_recommender(random_state: int = RANDOM_STATE) -> Callable[..., pd.DataFrame]:
    
    rng = np.random.default_rng(random_state)

    def recommend_random(user_id, train_ratings, movie_features, cold_start_movie_ids,
                         tfidf_matrix, movie_id_to_index, top_n=10):
        rated = set(train_ratings.loc[train_ratings["userID"] == user_id, "movieID"].tolist())
        cands = [m for m in dict.fromkeys(cold_start_movie_ids)
                 if m in movie_id_to_index and m not in rated]
        picked = rng.permutation(len(cands))[:top_n]
        rows = np.array([movie_id_to_index[cands[i]] for i in picked], dtype=int)
        top = movie_features.iloc[rows]
        return pd.DataFrame({"movieID": top["movieID"].to_numpy(), "title": top["title"].to_numpy(),
                             "recommendation_score": np.zeros(len(rows))})
    return recommend_random


# --------------------------------------------------------------------------
# 7. Evaluation
# --------------------------------------------------------------------------
def precision_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
   
    return sum(1 for m in list(recommended)[:k] if m in relevant) / k


def recall_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    
    return sum(1 for m in list(recommended)[:k] if m in relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    
    top = list(recommended)[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, m in enumerate(top) if m in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_item_recommendations(
    train_ratings: pd.DataFrame,
    heldout_ratings: pd.DataFrame,
    movie_features: pd.DataFrame,
    candidate_movie_ids: Iterable[int],
    tfidf_matrix: sparse.spmatrix,
    movie_id_to_index: Dict[int, int],
    k_values: Sequence[int] = (5, 10),
    relevance_threshold: float = POSITIVE_THRESHOLD,
    min_train_ratings: int = 10,
    min_positive_train: int = 5,
    recommender: Optional[Callable[..., pd.DataFrame]] = None,
    return_per_user: bool = False,
):
    
    recommender = recommender or recommend_cold_start_movies
    k_values = sorted(set(k_values))
    k_max = k_values[-1]
    candidates = sorted(set(int(m) for m in candidate_movie_ids))
    cand_set = set(candidates)

    held = heldout_ratings[heldout_ratings["movieID"].isin(cand_set)]
    relevant_by_user = (held[held["rating"] >= relevance_threshold]
                        .groupby("userID")["movieID"].agg(lambda s: set(s.tolist())))

    n_train = train_ratings.groupby("userID").size()
    n_pos = train_ratings[train_ratings["rating"] >= POSITIVE_THRESHOLD].groupby("userID").size()
    has_history = (n_train.reindex(relevant_by_user.index).fillna(0) >= min_train_ratings) & \
                  (n_pos.reindex(relevant_by_user.index).fillna(0) >= min_positive_train)
    users = relevant_by_user.index[has_history.to_numpy()]

    train_by_user = {u: g for u, g in train_ratings[train_ratings["userID"].isin(users)].groupby("userID")}

    per_user, recommended_union = [], set()
    for user in users:
        recs = recommender(user, train_by_user[user], movie_features, candidates,
                           tfidf_matrix, movie_id_to_index, top_n=k_max)
        if recs.empty:
            continue
        rec_ids = recs["movieID"].tolist()
        recommended_union.update(rec_ids[:k_max])
        relevant = relevant_by_user[user]
        row = {"userID": user, "n_relevant": len(relevant)}
        for k in k_values:
            row[f"Precision@{k}"] = precision_at_k(rec_ids, relevant, k)
            row[f"Recall@{k}"] = recall_at_k(rec_ids, relevant, k)
            row[f"NDCG@{k}"] = ndcg_at_k(rec_ids, relevant, k)
        per_user.append(row)

    per_user_df = pd.DataFrame(per_user)
    result = {
        "n_candidate_movies": len(candidates),
        "n_users_evaluated": len(per_user_df),
        "n_heldout_interactions": int(len(held)),
        "n_relevant_interactions": int((held["rating"] >= relevance_threshold).sum()),
    }
    for k in k_values:
        for metric in ("Precision", "Recall", "NDCG"):
            col = f"{metric}@{k}"
            result[col] = float(per_user_df[col].mean()) if len(per_user_df) else np.nan
    result[f"Catalog_Coverage@{k_max}"] = len(recommended_union) / len(candidates) if candidates else np.nan
    return (result, per_user_df) if return_per_user else result


# --------------------------------------------------------------------------
# 8. Item-similarity table (for item_similarity_results.csv) and plots
# --------------------------------------------------------------------------
def build_item_similarity_table(query_movie_ids: Iterable[int], movie_features: pd.DataFrame,
                                models: Dict[str, Tuple[str, sparse.spmatrix]],
                                movie_id_to_index: Dict[int, int], top_n: int = 10) -> pd.DataFrame:
   
    mf = movie_features.reset_index(drop=True)
    genre_sets = {int(m): set(str(g).lower().split()) if pd.notna(g) else set()
                  for m, g in zip(mf["movieID"], mf["genres"])}
    title_of = dict(zip(mf["movieID"].astype(int), mf["title"]))
    rows = []
    for name, (column, matrix) in models.items():
        for q in query_movie_ids:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                recs = recommend_similar_movies(q, mf, top_n, column, matrix, movie_id_to_index)
            for rank, rec in enumerate(recs.itertuples(index=False), start=1):
                a, b = genre_sets[int(q)], genre_sets[int(rec.movieID)]
                jac = len(a & b) / len(a | b) if (a | b) else np.nan
                rows.append({"model": name, "feature_column": column,
                             "query_movieID": int(q), "query_title": title_of[int(q)],
                             "rank": rank, "similar_movieID": int(rec.movieID),
                             "similar_title": rec.title,
                             "similarity_score": rec.similarity_score, "genre_jaccard": jac})
    return pd.DataFrame(rows)


def plot_item_cold_start_comparison(results: pd.DataFrame, save_path: Path,
                                    models: Sequence[str] = ("Genre TF-IDF", "Tags TF-IDF", "Combined TF-IDF"),
                                    metrics: Sequence[str] = ("Precision@5", "Recall@5", "NDCG@5")):
   
    import matplotlib
    import matplotlib.pyplot as plt

    categories = [c for c in CATEGORIES if c in set(results["category"])]
    titles = {CATEGORY_COLD: "Cold-start (0 training ratings)",
              CATEGORY_LIMITED: "Limited-rating (1-5)", CATEGORY_WARM: "Warm (> 5)"}
    colours = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, len(categories), figsize=(5.2 * len(categories), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / len(models)
    x = np.arange(len(metrics))
    for ax, cat in zip(axes, categories):
        sub = results[results["category"] == cat].set_index("model")
        for i, model in enumerate(models):
            vals = [sub.loc[model, m] for m in metrics]
            bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
                          label=model, color=colours[i])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(titles.get(cat, cat))
        ax.set_xlabel("Metric (top-5 recommendations)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean score over evaluated users")
    axes[-1].legend(title="Model", loc="upper right")
    fig.suptitle("Item Cold Start: Genre vs Tags vs Combined TF-IDF (held-out ratings >= 4.0 are relevant)")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metadata_feature_comparison(coverage: pd.DataFrame, save_path: Path):
    
    import matplotlib.pyplot as plt

    data = coverage[coverage["metadata_field"].isin(METADATA_FIELDS)]
    table = data.pivot(index="metadata_field", columns="scope", values="availability_pct")
    table = table.loc[METADATA_FIELDS, list(dict.fromkeys(data["scope"]))]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / table.shape[1]
    x = np.arange(len(table))
    for i, scope in enumerate(table.columns):
        bars = ax.bar(x + (i - (table.shape[1] - 1) / 2) * width, table[scope], width, label=scope)
        for b, v in zip(bars, table[scope]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in table.index])
    ax.set_ylim(0, 110)
    ax.set_xlabel("Metadata field")
    ax.set_ylabel("Movies with the field available (%)")
    ax.set_title("Metadata availability by field")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Movie group", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=table.shape[1])
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
