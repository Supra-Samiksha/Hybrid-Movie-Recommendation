import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. BUILD TF-IDF MODEL
# ============================================================

def build_tfidf_model(movie_features):

    required_columns = ["movieID", "title", "genres", "content"]

    for column in required_columns:
        if column not in movie_features.columns:
            raise ValueError(f"Missing required column: {column}")

    movie_features = movie_features.copy()
    movie_features = movie_features.drop_duplicates(
        subset="movieID"
    ).reset_index(drop=True)

    movie_features["content"] = (
        movie_features["content"].fillna("").astype(str)
    )

    tfidf = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b"
    )

    tfidf_matrix = tfidf.fit_transform(
        movie_features["content"]
    )

    return {
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "movie_features": movie_features
    }


# ============================================================
# 2. GENERATE RECOMMENDATIONS FOR A NEW USER
# ============================================================

def recommend_for_new_user(
    preferences,
    movie_features,
    top_n=10,
    user_id=None,
    exclude_movie_ids=None,
    model=None
):

    # Build the model if one was not supplied
    if model is None:
        model = build_tfidf_model(movie_features)

    tfidf = model["tfidf"]
    tfidf_matrix = model["tfidf_matrix"]
    movie_data = model["movie_features"]

    # Collect user preferences
    preference_tokens = []

    for category in ["genres", "actors", "directors"]:
        values = preferences.get(category, [])

        if isinstance(values, str):
            values = [values]

        preference_tokens.extend(
            str(value) for value in values
        )

    # Add favourite movie content
    favourite_movie_ids = preferences.get("movies", [])

    if isinstance(favourite_movie_ids, (int, np.integer)):
        favourite_movie_ids = [favourite_movie_ids]

    favourite_movie_ids = list(favourite_movie_ids)

    movie_lookup = movie_data.set_index("movieID")

    for movie_id in favourite_movie_ids:

        if movie_id in movie_lookup.index:

            movie_content = movie_lookup.loc[movie_id, "content"]

            if isinstance(movie_content, pd.Series):
                movie_content = movie_content.iloc[0]

            preference_tokens.append(str(movie_content))

    # Check whether preferences are available
    user_profile = " ".join(preference_tokens).strip()

    if not user_profile:
        return pd.DataFrame(
            columns=["movieID", "title", "genres", "similarity"]
        )

    # Convert user preferences into TF-IDF
    user_vector = tfidf.transform([user_profile])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    # Prepare recommendation results
    recommendations = movie_data[
        ["movieID", "title", "genres"]
    ].copy()

    recommendations["similarity"] = similarity_scores

    # Exclude onboarding movies and favourite movies
    excluded_ids = set(
        exclude_movie_ids or []
    )

    excluded_ids.update(favourite_movie_ids)

    recommendations = recommendations[
        ~recommendations["movieID"].isin(excluded_ids)
    ]

    # Sort by similarity
    recommendations = recommendations.sort_values(
        by="similarity",
        ascending=False
    )

    # Return Top-N recommendations
    return recommendations.head(top_n).reset_index(drop=True)


# ============================================================
# 3. PRECISION AT K
# ============================================================

def precision_at_k(recommended, relevant, k):

    recommended_k = recommended[:k]

    if not recommended_k:
        return 0.0

    hits = len(
        set(recommended_k) & set(relevant)
    )

    return hits / k


# ============================================================
# 4. RECALL AT K
# ============================================================

def recall_at_k(recommended, relevant, k):

    if not relevant:
        return 0.0

    recommended_k = recommended[:k]

    hits = len(
        set(recommended_k) & set(relevant)
    )

    return hits / len(relevant)


# ============================================================
# 5. NDCG AT K
# ============================================================

def ndcg_at_k(recommended, relevant, k):

    recommended_k = recommended[:k]

    relevance = [
        1 if movie in relevant else 0
        for movie in recommended_k
    ]

    dcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(relevance)
    )

    ideal_relevance = [
        1
    ] * min(len(relevant), k)

    idcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(ideal_relevance)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# 6. CATALOG COVERAGE
# ============================================================

def catalog_coverage(recommended_movie_ids, catalog_size):

    if catalog_size == 0:
        return 0.0

    unique_movies = set(recommended_movie_ids)

    return len(unique_movies) / catalog_size