import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


data_folder = Path("../dataset/processed")
output_folder = Path("../outputs")
output_folder.mkdir(parents=True, exist_ok=True)

train_path = data_folder / "train_ratings.csv"
test_path = data_folder / "test_ratings.csv"

if not train_path.exists():
    raise FileNotFoundError(f"Training file not found: {train_path}")
if not test_path.exists():
    raise FileNotFoundError(
        f"Test file not found: {test_path}. Place the team's shared test_ratings.csv "
        "inside dataset/processed before running the full evaluation."
    )

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

required_columns = {"userID", "movieID", "rating"}

for name, frame in [("train", train), ("test", test)]:
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")

    if frame[list(required_columns)].isnull().any().any():
        raise ValueError(f"{name} has missing userID, movieID or rating values.")

    if frame.duplicated(["userID", "movieID"]).any():
        raise ValueError(f"{name} contains duplicate userID/movieID pairs.")

print(f"Training ratings: {len(train):,}")
print(f"Testing ratings: {len(test):,}")
print("Columns:", list(train.columns))
print(f"Training users: {train['userID'].nunique():,}")
print(f"Training movies: {train['movieID'].nunique():,}")
print(f"Rating range: {train['rating'].min():g} to {train['rating'].max():g}")


all_ids = pd.concat(
    [train[["userID", "movieID"]], test[["userID", "movieID"]]],
    ignore_index=True
)

user_ids = np.sort(all_ids["userID"].unique())
movie_ids = np.sort(all_ids["movieID"].unique())

n_users = len(user_ids)
n_movies = len(movie_ids)
total_possible = n_users * n_movies
total_recorded = len(train) + len(test)

density = total_recorded / total_possible * 100
overall_sparsity = 100 - density

print(f"Unique users: {n_users:,}")
print(f"Unique movies: {n_movies:,}")
print(f"Possible user-movie pairs: {total_possible:,}")
print(f"Recorded ratings: {total_recorded:,}")
print(f"Matrix density: {density:.4f}%")
print(f"Matrix sparsity: {overall_sparsity:.4f}%")


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

train["rating"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], title="Training ratings by score"
)
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Number of ratings")

ratings_per_user = train.groupby("userID").size()
ratings_per_movie = train.groupby("movieID").size()

axes[1].hist(ratings_per_user, bins=30)
axes[1].set_title("Ratings per user")
axes[1].set_xlabel("Number of ratings")
axes[1].set_ylabel("Number of users")

axes[2].hist(ratings_per_movie, bins=30)
axes[2].set_title("Ratings per movie")
axes[2].set_xlabel("Number of ratings")
axes[2].set_ylabel("Number of movies")

plt.tight_layout()
plt.savefig(output_folder / "sparsity_distribution.png", dpi=150)
plt.show()

print("Median ratings per user:", ratings_per_user.median())
print("Median ratings per movie:", ratings_per_movie.median())


availability_levels = [100, 75, 50, 25, 10]
shuffled_train = train.sample(frac=1, random_state=42).reset_index(drop=True)

sparsity_rows = []
for percent in availability_levels:
    sample_size = int(len(shuffled_train) * percent / 100)
    sample = shuffled_train.iloc[:sample_size]

    sample_density = len(sample) / total_possible * 100
    sparsity_rows.append({
        "Data Availability (%)": percent,
        "Training Ratings": len(sample),
        "Matrix Density (%)": sample_density,
        "Matrix Sparsity (%)": 100 - sample_density
    })

sparsity_df = pd.DataFrame(sparsity_rows)
display(sparsity_df.round(4))
sparsity_df.to_csv(output_folder / "sparsity_results.csv", index=False)

plt.figure(figsize=(7, 4))
plt.plot(
    sparsity_df["Data Availability (%)"],
    sparsity_df["Matrix Sparsity (%)"],
    marker="o"
)
plt.xlabel("Available training ratings (%)")
plt.ylabel("Unobserved matrix cells (%)")
plt.title("Matrix sparsity at each training-data level")
plt.gca().invert_xaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_folder / "sparsity_vs_availability.png", dpi=150)
plt.show()


user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
movie_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}

def to_arrays(frame):
    user_index = frame["userID"].map(user_to_index).to_numpy(dtype=int)
    movie_index = frame["movieID"].map(movie_to_index).to_numpy(dtype=int)
    ratings = frame["rating"].to_numpy(dtype=float)
    return user_index, movie_index, ratings

test_u, test_m, test_r = to_arrays(test)

rating_min = float(train["rating"].min())
rating_max = float(train["rating"].max())

print(f"Matrix shape: {n_users} users × {n_movies} movies")
print(f"Rating scale in training data: {rating_min:g} to {rating_max:g}")


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))

def fit_biases(u, m, r, reg_user=15, reg_item=10):
    # Overall average rating.
    global_mean = float(np.mean(r))

    # How many ratings each movie / user has.
    item_count = np.bincount(m, minlength=n_movies)
    user_count = np.bincount(u, minlength=n_users)

    # Movie bias: how far above / below average a movie is rated.
    # Adding reg_item to the count shrinks the bias towards 0 for movies with few ratings.
    item_sum = np.bincount(m, weights=r - global_mean, minlength=n_movies)
    item_bias = item_sum / (item_count + reg_item)

    # User bias: how generous a user is, after removing the movie effect.
    user_residual = r - global_mean - item_bias[m]
    user_sum = np.bincount(u, weights=user_residual, minlength=n_users)
    user_bias = user_sum / (user_count + reg_user)

    return global_mean, user_bias, item_bias

def predict_global(global_mean, number_of_rows):
    return np.full(number_of_rows, global_mean, dtype=float)

def predict_bias(global_mean, user_bias, item_bias, u, m):
    estimates = global_mean + user_bias[u] + item_bias[m]
    return np.clip(estimates, rating_min, rating_max)


NEIGHBOURS = 20

def build_rating_matrix(u, m, r):
    # Only observed ratings are stored. An absent cell is not a zero rating.
    return csr_matrix((r, (u, m)), shape=(n_users, n_movies))

def fit_neighbour_model(u, m, r, mode="user", neighbours=NEIGHBOURS):
    rating_matrix = build_rating_matrix(u, m, r)

    # "Entities" are the things we compare: users (rows) or movies (columns).
    # For item-based CF we transpose so each row is one movie.
    if mode == "user":
        entity_matrix = rating_matrix
    elif mode == "item":
        entity_matrix = rating_matrix.T.tocsr()
    else:
        raise ValueError("mode must be 'user' or 'item'")

    # +1 because every entity's nearest neighbour is itself; we drop it later.
    n_neighbours = min(neighbours + 1, entity_matrix.shape[0])

    model = NearestNeighbors(n_neighbors=n_neighbours, metric="cosine", algorithm="brute")
    model.fit(entity_matrix)

    distances, indices = model.kneighbors(entity_matrix)
    similarities = 1 - distances          # cosine similarity = 1 - cosine distance

    fallback = float(entity_matrix.data.mean()) if entity_matrix.nnz else 0.0
    return entity_matrix, indices, similarities, fallback

def predict_neighbourhood(model, query_u, query_m, mode="user"):
    entity_matrix, neighbour_ids, similarities, fallback = model
    predictions = np.empty(len(query_u), dtype=float)

    for row, (user, movie) in enumerate(zip(query_u, query_m)):
        # User-based: look at similar USERS and what they gave THIS movie.
        # Item-based: look at similar MOVIES and what THIS user gave them.
        # Both cases are the same lookup once the matrix rows are the entities.
        if mode == "user":
            entity, target = user, movie
        else:
            entity, target = movie, user

        ids = neighbour_ids[entity]
        sims = similarities[entity]
        keep_neighbour = ids != entity            # remove the entity itself
        ids, sims = ids[keep_neighbour], sims[keep_neighbour]

        neighbour_ratings = entity_matrix[ids, target].toarray().ravel()
        rated = neighbour_ratings > 0              # which neighbours actually rated it
        weights = np.maximum(sims[rated], 0)       # ignore negative similarities

        if rated.any() and weights.sum() > 0:
            # Similarity-weighted average of the neighbours' ratings.
            predictions[row] = np.dot(weights, neighbour_ratings[rated]) / weights.sum()
        else:
            predictions[row] = fallback

    return np.clip(predictions, rating_min, rating_max)


def fit_svd_style(u, m, r, k=20, epochs=10, learning_rate=0.01, reg=0.05, seed=42):
    global_mean, user_bias, item_bias = fit_biases(u, m, r)

    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, size=(n_users, k))
    Q = rng.normal(0, 0.05, size=(n_movies, k))

    order = np.arange(len(r))

    for epoch in range(epochs):
        rng.shuffle(order)
        for index in order:
            user = u[index]
            movie = m[index]
            actual = r[index]

            user_vector = P[user]
            item_vector = Q[movie]

            prediction = (
                global_mean + user_bias[user] + item_bias[movie]
                + np.dot(user_vector, item_vector)
            )
            error = actual - prediction

            # Standard FunkSVD SGD step. Biases come from fit_biases, so
            # only the factor vectors are updated here.
            P[user]  = user_vector + learning_rate * (error * item_vector - reg * user_vector)
            Q[movie] = item_vector + learning_rate * (error * user_vector - reg * item_vector)

    # Users / movies that never appeared in the training sample still have
    # their random init values. Zero them out so they don't inject noise.
    seen_users = np.unique(u)
    seen_movies = np.unique(m)
    P[np.setdiff1d(np.arange(n_users), seen_users)] = 0.0
    Q[np.setdiff1d(np.arange(n_movies), seen_movies)] = 0.0

    return global_mean, user_bias, item_bias, P, Q

def predict_factor_model(model, u, m):
    global_mean, user_bias, item_bias, P, Q = model
    estimates = (
        global_mean + user_bias[u] + item_bias[m]
        + np.sum(P[u] * Q[m], axis=1)
    )
    return np.clip(estimates, rating_min, rating_max)


def als_update(matrix, fixed, lam):
    """Re-solve every row of one factor matrix while the other one stays fixed.

    matrix : sparse (entities x others) matrix of residual ratings, in CSR format
    fixed  : factor matrix of the "others" (held constant)
    Returns the new factor matrix for the entities.
    """
    k = fixed.shape[1]
    new_factors = np.zeros((matrix.shape[0], k))   # rows with no ratings stay zero

    for row in range(matrix.shape[0]):
        # In CSR format, indptr[row] : indptr[row + 1] marks where this
        # row's observed ratings sit inside the `indices` and `data` arrays.
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if start == end:
            continue                               # nothing observed for this row

        other_ids = matrix.indices[start:end]      # which movies (or users) were rated
        values = matrix.data[start:end]            # the ratings themselves
        F = fixed[other_ids]                       # their factor rows

        A = F.T @ F + lam * len(values) * np.eye(k)
        b = F.T @ values
        new_factors[row] = np.linalg.solve(A, b)   # solve A p = b

    return new_factors

def fit_als(u, m, r, k=20, lam=0.1, iterations=10, seed=42):
    global_mean, user_bias, item_bias = fit_biases(u, m, r)

    # Factor model only has to explain what the biases could not.
    residual = r - global_mean - user_bias[u] - item_bias[m]

    user_matrix = csr_matrix((residual, (u, m)), shape=(n_users, n_movies))
    movie_matrix = user_matrix.T.tocsr()           # same data, one row per movie

    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, size=(n_users, k))     # random start (P is overwritten in the first step)
    Q = rng.normal(0, 0.05, size=(n_movies, k))    # random start for Q; P is solved from it first

    for _ in range(iterations):
        P = als_update(user_matrix, Q, lam)        # users, with movies fixed
        Q = als_update(movie_matrix, P, lam)       # movies, with users fixed

    return global_mean, user_bias, item_bias, P, Q


K_VALUES = [10, 20]
ALS_LAM_VALUES = [0.1]

TUNE_EPOCHS = 5      # SGD passes / ALS iterations while tuning (kept short)
FINAL_EPOCHS = 10    # SGD passes / ALS iterations for the final models

def tune_factor_models(u, m, r):
    # Split this sample 90% fit / 10% validation.
    rng = np.random.default_rng(123)
    order = rng.permutation(len(r))
    split = int(0.9 * len(r))
    fit_indices, val_indices = order[:split], order[split:]

    if len(val_indices) == 0:
        raise ValueError("Not enough training ratings to create a validation split.")

    fit_u, fit_m, fit_r = u[fit_indices], m[fit_indices], r[fit_indices]
    val_u, val_m, val_r = u[val_indices], m[val_indices], r[val_indices]

    records = []

    for k in K_VALUES:
        svd_model = fit_svd_style(fit_u, fit_m, fit_r, k=k, epochs=TUNE_EPOCHS, seed=42)
        svd_prediction = predict_factor_model(svd_model, val_u, val_m)
        records.append({
            "Model": "SVD-style", "K": k, "Lambda": np.nan,
            "Validation RMSE": rmse(val_r, svd_prediction)
        })

        for lam in ALS_LAM_VALUES:
            als_model = fit_als(fit_u, fit_m, fit_r, k=k, lam=lam, iterations=TUNE_EPOCHS)
            als_prediction = predict_factor_model(als_model, val_u, val_m)
            records.append({
                "Model": "ALS", "K": k, "Lambda": lam,
                "Validation RMSE": rmse(val_r, als_prediction)
            })

    results = pd.DataFrame(records)
    best_svd = results[results["Model"] == "SVD-style"].sort_values("Validation RMSE").iloc[0]
    best_als = results[results["Model"] == "ALS"].sort_values("Validation RMSE").iloc[0]

    return int(best_svd["K"]), int(best_als["K"]), float(best_als["Lambda"]), results


def evaluate_level(percent):
    sample_size = int(len(shuffled_train) * percent / 100)
    sample = shuffled_train.iloc[:sample_size]
    u, m, r = to_arrays(sample)

    # Baselines
    global_mean, user_bias, item_bias = fit_biases(u, m, r)
    global_prediction = predict_global(global_mean, len(test_r))
    bias_prediction = predict_bias(global_mean, user_bias, item_bias, test_u, test_m)

    # Neighbourhood models
    start = time.perf_counter()
    user_model = fit_neighbour_model(u, m, r, mode="user")
    user_cf_prediction = predict_neighbourhood(user_model, test_u, test_m, mode="user")
    user_cf_time = time.perf_counter() - start

    start = time.perf_counter()
    item_model = fit_neighbour_model(u, m, r, mode="item")
    item_cf_prediction = predict_neighbourhood(item_model, test_u, test_m, mode="item")
    item_cf_time = time.perf_counter() - start

    # Select factor-model settings using only this sample's validation split.
    best_svd_k, best_als_k, best_als_lam, tuning_results = tune_factor_models(u, m, r)

    start = time.perf_counter()
    svd_model = fit_svd_style(u, m, r, k=best_svd_k, epochs=FINAL_EPOCHS)
    svd_time = time.perf_counter() - start
    svd_prediction = predict_factor_model(svd_model, test_u, test_m)

    start = time.perf_counter()
    als_model = fit_als(u, m, r, k=best_als_k, lam=best_als_lam, iterations=FINAL_EPOCHS)
    als_time = time.perf_counter() - start
    als_prediction = predict_factor_model(als_model, test_u, test_m)

    predictions = {
        "Global Mean": global_prediction,
        "Bias": bias_prediction,
        "User CF": user_cf_prediction,
        "Item CF": item_cf_prediction,
        "SVD-style": svd_prediction,
        "ALS": als_prediction
    }

    # Sanity checks: catch silent bugs such as NaN or out-of-range predictions.
    for name, p in predictions.items():
        assert len(p) == len(test_r), f"{name}: length mismatch"
        assert not np.isnan(p).any(), f"{name}: contains NaN"
        assert p.min() >= rating_min - 1e-9 and p.max() <= rating_max + 1e-9, \
            f"{name}: predictions outside [{rating_min}, {rating_max}]"

    row = {
        "Data Availability (%)": percent,
        "Training Ratings": len(sample),
        # total_possible uses the train+test ID universe, the same definition at every level.
        "Matrix Sparsity (%)": 100 - len(sample) / total_possible * 100,
        "SVD K": best_svd_k,
        "ALS K": best_als_k,
        "ALS Lambda": best_als_lam,
        "User CF time (s)": user_cf_time,
        "Item CF time (s)": item_cf_time,
        "SVD-style time (s)": svd_time,
        "ALS time (s)": als_time
    }

    for name, prediction in predictions.items():
        row[name + " RMSE"] = rmse(test_r, prediction)
        row[name + " MAE"] = mae(test_r, prediction)

    return row, tuning_results

comparison_rows = []
tuning_tables = {}

for percent in availability_levels:
    print(f"Running experiment with {percent}% of training ratings...")
    result, tuning_table = evaluate_level(percent)
    comparison_rows.append(result)
    tuning_tables[percent] = tuning_table

model_results = pd.DataFrame(comparison_rows)
display(model_results.round(4))


model_results.to_csv(output_folder / "model_comparison.csv", index=False)

# Validation choices at each level
display(model_results[["Data Availability (%)", "SVD K", "ALS K", "ALS Lambda"]])

metric_names = ["Global Mean", "Bias", "User CF", "Item CF", "SVD-style", "ALS"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for name in metric_names:
    axes[0].plot(model_results["Data Availability (%)"],
                 model_results[name + " RMSE"], marker="o", label=name)
    axes[1].plot(model_results["Data Availability (%)"],
                 model_results[name + " MAE"], marker="o", label=name)

axes[0].set_title("RMSE vs training-data availability")
axes[0].set_xlabel("Training data available (%)")
axes[0].set_ylabel("RMSE (lower is better)")
axes[1].set_title("MAE vs training-data availability")
axes[1].set_xlabel("Training data available (%)")
axes[1].set_ylabel("MAE (lower is better)")

for axis in axes:
    axis.invert_xaxis()
    axis.grid(True)
    axis.legend()

plt.tight_layout()
plt.savefig(output_folder / "algorithm_comparison.png", dpi=150)
plt.show()
