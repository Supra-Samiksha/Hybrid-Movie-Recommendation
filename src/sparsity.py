import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Set the locations of the dataset files
data_folder = Path("dataset/processed")

# Step 2: Read the training and testing datasets
train = pd.read_csv(data_folder / "train_ratings.csv")
test = pd.read_csv(data_folder / "test_ratings.csv")

print("Training and testing datasets loaded successfully!")

# Step 3: Combine both files only to identify all users and movies
# The test ratings will not be modified or used in the experiments.
all_ratings = pd.concat([train, test], ignore_index=True)

total_users = all_ratings["userID"].nunique()
total_movies = all_ratings["movieID"].nunique()

# Calculate all possible user-movie combinations
total_possible = total_users * total_movies

# Calculate the sparsity of the complete dataset
total_ratings = len(all_ratings)

density = (total_ratings / total_possible) * 100
sparsity = 100 - density

print("\nDATASET INFORMATION")
print("-------------------")
print("Total Users:", total_users)
print("Total Movies:", total_movies)
print("Total Ratings:", total_ratings)
print("Training Ratings:", len(train))
print("Testing Ratings:", len(test))
print("Matrix Density:", round(density, 2), "%")
print("Matrix Sparsity:", round(sparsity, 2), "%")

# Step 4: Test different levels of training data availability
# The test dataset remains unchanged throughout the experiment.

availability_levels = [100, 75, 50, 25, 10]

# Shuffle the training data once so that all experiments
# use consistent and repeatable samples.
shuffled_train = train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

results = []

for percentage in availability_levels:

    # Select the required percentage of training ratings
    number_of_ratings = int(len(shuffled_train) * percentage / 100)

    selected_train = shuffled_train.iloc[:number_of_ratings]

    # Calculate density and sparsity using fixed matrix dimensions
    current_density = (
        len(selected_train) / total_possible
    ) * 100

    current_sparsity = 100 - current_density

    results.append({
        "Data Availability (%)": percentage,
        "Training Ratings": len(selected_train),
        "Test Ratings": len(test),
        "Matrix Density (%)": round(current_density, 4),
        "Matrix Sparsity (%)": round(current_sparsity, 4)
    })

# Step 5: Display the experiment results
results_df = pd.DataFrame(results)

print("\nSPARSITY AT DIFFERENT DATA LEVELS")
print("---------------------------------")
print(results_df.to_string(index=False))

# Step 6: Save the results
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

results_df.to_csv(
    output_folder / "sparsity_results.csv",
    index=False
)

# Step 7: Generate the sparsity graph
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Data Availability (%)"],
    results_df["Matrix Sparsity (%)"],
    marker="o"
)

plt.xlabel("Training Data Availability (%)")
plt.ylabel("Matrix Sparsity (%)")
plt.title("Effect of Data Availability on Matrix Sparsity")

plt.gca().invert_xaxis()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    output_folder / "sparsity_vs_availability.png"
)

plt.show()

print("\nResults and graph saved successfully!")
