
import numpy as np
import pandas as pd


def get_sparsity_aware_weights(user_rating_count):
    """
    Select recommendation weights based on the amount of
    rating history available for the current user.

    Users with fewer interactions rely more on content-based
    recommendations. Users with richer histories receive
    greater collaborative-filtering weight.
    """

    if user_rating_count < 50:
        return (0.10, 0.70, 0.20)

    elif user_rating_count < 100:
        return (0.20, 0.60, 0.20)

    elif user_rating_count < 300:
        return (0.40, 0.40, 0.20)

    elif user_rating_count < 500:
        return (0.55, 0.30, 0.15)

    else:
        return (0.65, 0.25, 0.10)


def normalize_scores(recommendations, score_column):
    """
    Normalize recommendation scores to the range [0, 1].

    If all scores are identical, assign 1.0 to each recommendation
    so that a valid recommendation list is not discarded.
    """

    if recommendations is None or recommendations.empty:
        return pd.DataFrame(columns=["movieID", "normalized_score"])

    if "movieID" not in recommendations.columns:
        raise ValueError("Recommendations must contain movieID.")

    if score_column not in recommendations.columns:
        raise ValueError(f"Missing score column: {score_column}")

    result = recommendations[["movieID", score_column]].copy()

    result[score_column] = pd.to_numeric(
        result[score_column], errors="coerce"
    )

    result = result.dropna(subset=["movieID", score_column])
    result = result.drop_duplicates("movieID")

    if result.empty:
        return pd.DataFrame(columns=["movieID", "normalized_score"])

    minimum = result[score_column].min()
    maximum = result[score_column].max()

    if maximum == minimum:
        result["normalized_score"] = 1.0
    else:
        result["normalized_score"] = (
            (result[score_column] - minimum) / (maximum - minimum)
        )

    return result[["movieID", "normalized_score"]]


def combine_recommendations(
    cf_recommendations=None,
    content_recommendations=None,
    item_recommendations=None,
    weights=(0.5, 0.3, 0.2),
    top_n=10,
    exclude_movie_ids=None
):
    """
    Combine recommendations from three components:

    1. System Cold Start: Collaborative Filtering
    2. User Cold Start: User preference-based content filtering
    3. Item Cold Start: Metadata-based cold-item recommendations

    Scores are normalized independently before weighted fusion.
    Missing component scores are excluded from the denominator.
    """

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    if len(weights) != 3 or any(w < 0 for w in weights):
        raise ValueError("Provide three non-negative weights.")

    if sum(weights) <= 0:
        raise ValueError("At least one weight must be positive.")

    sources = [
        (cf_recommendations, "predicted_rating", weights[0], "cf_score"),
        (content_recommendations, "similarity", weights[1], "content_score"),
        (item_recommendations, "recommendation_score", weights[2], "item_score")
    ]

    combined = None

    # Normalize each component's scores and merge by movieID.
    for recommendations, score_column, weight, output_column in sources:

        if weight == 0 or recommendations is None or recommendations.empty:
            continue

        normalized = normalize_scores(recommendations, score_column)

        if normalized.empty:
            continue

        normalized = normalized.rename(
            columns={"normalized_score": output_column}
        )

        # Retain movie details for the final output.
        details = [
            column for column in ["movieID", "title", "genres"]
            if column in recommendations.columns
        ]

        details_df = recommendations[details].drop_duplicates("movieID")

        normalized = normalized.merge(
            details_df,
            on="movieID",
            how="left"
        )

        if combined is None:
            combined = normalized.copy()
        else:
            combined = combined.merge(
                normalized,
                on="movieID",
                how="outer",
                suffixes=("", "_new")
            )

            # Fill missing movie details from the newly merged source.
            for column in ["title", "genres"]:
                new_column = f"{column}_new"

                if new_column in combined.columns:
                    if column not in combined.columns:
                        combined[column] = combined[new_column]
                    else:
                        combined[column] = combined[column].combine_first(
                            combined[new_column]
                        )

                    combined.drop(columns=[new_column], inplace=True)

    if combined is None or combined.empty:
        return pd.DataFrame(
            columns=[
                "movieID", "title", "genres", "hybrid_score",
                "cf_score", "content_score", "item_score"
            ]
        )

    # Calculate the weighted average using only available components.
    numerator = np.zeros(len(combined), dtype=float)
    denominator = np.zeros(len(combined), dtype=float)

    score_columns = [
        ("cf_score", weights[0]),
        ("content_score", weights[1]),
        ("item_score", weights[2])
    ]

    for column, weight in score_columns:

        if column not in combined.columns:
            combined[column] = np.nan

        available = combined[column].notna().to_numpy()

        numerator += (
            combined[column].fillna(0).to_numpy(dtype=float) * weight
        )

        denominator += available.astype(float) * weight

    combined["hybrid_score"] = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0
    )

    # Exclude movies already rated or otherwise unavailable.
    if exclude_movie_ids is not None:
        combined = combined[
            ~combined["movieID"].isin(set(exclude_movie_ids))
        ]

    combined = combined.sort_values(
        "hybrid_score",
        ascending=False
    )

    return combined[
        [
            "movieID", "title", "genres", "hybrid_score",
            "cf_score", "content_score", "item_score"
        ]
    ].head(top_n).reset_index(drop=True)
def get_sparsity_aware_weights(user_rating_count):
    """
    Return recommendation weights based on user rating history.
    """

    if user_rating_count < 50:
        return (0.10, 0.70, 0.20)

    elif user_rating_count < 100:
        return (0.20, 0.60, 0.20)

    elif user_rating_count < 300:
        return (0.40, 0.40, 0.20)

    elif user_rating_count < 500:
        return (0.55, 0.30, 0.15)

    else:
        return (0.65, 0.25, 0.10)