
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def train_stage_model(stage_data):
    """
    Build a mean-centred user-item model for one platform growth stage.

    Parameters
    ----------
    stage_data : DataFrame
        Must contain userID, movieID and rating.

    Returns
    -------
    dict
        Sparse rating matrices, mappings and user means.
    """
    required = {"userID", "movieID", "rating"}
    missing = required - set(stage_data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if stage_data.empty:
        raise ValueError("Cannot train a model with no interactions.")

    data = stage_data.dropna(
        subset=["userID", "movieID", "rating"]
    ).copy()

    if data.empty:
        raise ValueError("No valid interactions available.")

    if data.duplicated(["userID", "movieID"]).any():
        raise ValueError(
            "Duplicate user-movie interactions found. "
            "Aggregate or remove duplicates before training."
        )

    user_ids = data["userID"].unique()
    movie_ids = data["movieID"].unique()

    user_to_index = {
        uid: idx for idx, uid in enumerate(user_ids)
    }

    movie_to_index = {
        mid: idx for idx, mid in enumerate(movie_ids)
    }

    rows = data["userID"].map(user_to_index).to_numpy()
    cols = data["movieID"].map(movie_to_index).to_numpy()

    user_means = data.groupby("userID")["rating"].mean()

    centered_ratings = (
        data["rating"].to_numpy(dtype=float)
        - data["userID"].map(user_means).to_numpy(dtype=float)
    )

    centered_matrix = csr_matrix(
        (centered_ratings, (rows, cols)),
        shape=(len(user_ids), len(movie_ids))
    )

    raw_matrix = csr_matrix(
        (
            data["rating"].to_numpy(dtype=float),
            (rows, cols)
        ),
        shape=(len(user_ids), len(movie_ids))
    )

    return {
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_to_index": user_to_index,
        "movie_to_index": movie_to_index,
        "user_means": user_means,
        "centered_matrix": centered_matrix,
        "raw_matrix": raw_matrix
    }


def recommend_stage_cf(
    model,
    user_id,
    movie_features,
    top_n=10,
    k=20,
    min_support=2
):
    """
    Generate recommendations using mean-centred User-Based CF.

    Only movies not already rated by the target user are considered.
    """

    if top_n < 1 or k < 1 or min_support < 1:
        raise ValueError("top_n, k and min_support must be positive.")

    if user_id not in model["user_to_index"]:
        return pd.DataFrame()

    user_idx = model["user_to_index"][user_id]

    similarities = cosine_similarity(
        model["centered_matrix"][user_idx],
        model["centered_matrix"]
    ).flatten()

    similarities[user_idx] = 0

    neighbor_indices = np.argsort(similarities)[::-1]

    neighbor_indices = [
        idx for idx in neighbor_indices
        if similarities[idx] > 0
    ][:k]

    if not neighbor_indices:
        return pd.DataFrame()

    neighbor_similarities = similarities[neighbor_indices]

    neighbor_raw = model["raw_matrix"][
        neighbor_indices
    ].tocsr()

    neighbor_centered = neighbor_raw.astype(float).copy()

    # Centre each neighbour's ratings around their own mean.
    for i, neighbor_idx in enumerate(neighbor_indices):

        neighbor_id = model["user_ids"][neighbor_idx]
        mean_rating = model["user_means"].loc[neighbor_id]

        start = neighbor_centered.indptr[i]
        end = neighbor_centered.indptr[i + 1]

        neighbor_centered.data[start:end] -= mean_rating

    weighted_deviations = np.asarray(
        neighbor_similarities.reshape(1, -1)
        @ neighbor_centered
    ).flatten()

    weighted_counts = np.asarray(
        neighbor_similarities.reshape(1, -1)
        @ (neighbor_raw != 0)
    ).flatten()

    support = np.asarray(
        (neighbor_raw != 0).sum(axis=0)
    ).flatten()

    target_mean = model["user_means"].loc[user_id]

    raw_predictions = np.divide(
        weighted_deviations,
        weighted_counts,
        out=np.zeros_like(weighted_deviations, dtype=float),
        where=weighted_counts > 0
    ) + target_mean

    target_rated = (
        model["raw_matrix"][user_idx].toarray().flatten() != 0
    )

    valid = (
        (weighted_counts > 0)
        & (support >= min_support)
        & (~target_rated)
    )

    candidates = np.where(valid)[0]

    if len(candidates) == 0:
        return pd.DataFrame()

    ranked = candidates[
        np.argsort(raw_predictions[candidates])[::-1]
    ][:top_n]

    recommendations = pd.DataFrame({
        "movieID": model["movie_ids"][ranked],
        "raw_prediction": raw_predictions[ranked],
        "predicted_rating": np.clip(
            raw_predictions[ranked], 0.5, 5.0
        ),
        "neighbor_support": support[ranked]
    })

    recommendations = recommendations.merge(
        movie_features[
            ["movieID", "title", "genres"]
        ],
        on="movieID",
        how="left"
    )

    return recommendations[
        [
            "movieID",
            "title",
            "genres",
            "raw_prediction",
            "predicted_rating",
            "neighbor_support"
        ]
    ]
