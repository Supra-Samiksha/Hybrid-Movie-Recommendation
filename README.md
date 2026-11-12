# Hybrid Movie Recommendation System

A Hybrid Movie Recommendation System designed to address the major challenges faced by recommendation platforms, particularly **Data Sparsity, User Cold Start, Item Cold Start, and System Cold Start**.

The project combines Collaborative Filtering, Content-Based Filtering, Matrix Factorization, TF-IDF, and K-Means Clustering to generate personalised movie recommendations under different data availability conditions.

---

## 1. Project Overview

Recommendation systems help users discover relevant items from large collections of available content. However, their effectiveness depends heavily on the availability of sufficient user interaction data.

Traditional collaborative filtering approaches often struggle when interaction data is limited or unavailable. This project investigates these limitations through four major challenges and develops a hybrid recommendation framework that combines user preferences, movie metadata, and interaction patterns.

The system is developed using the HetRec 2011 – MovieLens 2K dataset.

## 2. Research Objectives

The primary objectives of this project are:

- To investigate the impact of data sparsity on recommendation quality.
- To develop recommendation strategies for new users with no historical ratings.
- To recommend newly introduced movies with limited or no rating history.
- To simulate a newly launched recommendation platform with zero initial interactions.
- To integrate collaborative and content-based recommendation approaches.
- To evaluate recommendation performance under different data availability conditions.

## 3. Research Challenges and Team Division

The project is divided into four independent research modules, each addressing a specific recommendation-system challenge.

| Member | Research Area | Primary Responsibility |
|---|---|---|
| Member 1 | Data Sparsity | Analyse sparse user-item interactions and implement SVD and ALS. |
| Member 2 | User Cold Start | Generate recommendations for new users using onboarding preferences and content-based filtering. |
| Member 3 | Item Cold Start | Recommend newly introduced movies using available metadata. |
| Member 4 | System Cold Start | Simulate a new recommendation platform and progressively introduce interaction data. |

All four modules use common preprocessing outputs and contribute to the final hybrid recommendation framework.

## 4. Dataset

**Dataset:** HetRec 2011 – MovieLens 2K

**Source:** https://grouplens.org/datasets/hetrec-2011/

The dataset contains movie ratings, user interactions, and associated movie metadata.

### Main Dataset Files

| File | Description |
|---|---|
| `user_ratedmovies.dat` | User movie ratings and timestamps |
| `movies.dat` | Movie information and external ratings |
| `movie_genres.dat` | Movie genre information |
| `movie_tags.dat` | Movie-tag associations and weights |
| `tags.dat` | Tag ID and tag-name mapping |
| `movie_actors.dat` | Movie cast information |
| `movie_directors.dat` | Movie director information |
| `movie_countries.dat` | Movie country information |

### Initial Dataset Statistics

| Parameter | Value |
|---|---:|
| Unique users | 2,113 |
| Movies with ratings | 10,109 |
| Total rating interactions | 855,598 |
| Rating scale | 0.5–5.0 |
| Average rating | 3.44 |
| User-item matrix sparsity | 95.99% |

The original dataset files are maintained separately from the processed datasets.

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook, VS Code |
| Data Processing | Pandas, NumPy |
| Scientific Computing | SciPy |
| Machine Learning | Scikit-learn |
| Recommendation Techniques | Collaborative Filtering, SVD, ALS |
| Content-Based Filtering | TF-IDF, Cosine Similarity |
| Clustering | K-Means |
| Data Visualisation | Matplotlib, Seaborn |
| Model Storage | Joblib |
| Version Control | Git, GitHub |

## 6. Recommendation Techniques

The project explores the following techniques:

### Collaborative Filtering
Uses historical user-item interactions to identify similarities between users or movies and generate recommendations.

### Matrix Factorization
- Singular Value Decomposition (SVD)
- Alternating Least Squares (ALS)

These techniques learn latent user and movie representations from rating interactions.

### Content-Based Filtering
Uses movie attributes such as genres, tags, actors, directors, and countries to identify similar movies.

### TF-IDF
Transforms textual movie metadata into numerical feature representations for similarity-based recommendations.

### K-Means Clustering
Groups movies or relevant feature representations into clusters to support similarity discovery and recommendation generation.

### Hybrid Recommendation
Combines collaborative and content-based recommendation scores to improve recommendation quality and address limitations of individual approaches.

## 7. Project Structure

```text
Hybrid-Movie-Recommendation-System/
│
├── dataset/
│   ├── raw/
│   │   └── Original dataset files (.dat)
│   │
│   └── processed/
│       ├── ratings_clean.csv
│       ├── movie_features.csv
│       ├── train_ratings.csv
│       └── test_ratings.csv
│
├── notebooks/
│   ├── 00_common_preprocessing.ipynb
│   ├── 01_data_sparsity.ipynb
│   ├── 02_user_cold_start.ipynb
│   ├── 03_item_cold_start.ipynb
│   ├── 04_system_cold_start.ipynb
│   └── 05_hybrid_integration.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── sparsity.py
│   ├── user_cold_start.py
│   ├── item_cold_start.py
│   ├── system_cold_start.py
│   └── hybrid.py
│
├── models/
│
├── outputs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure represents the planned organisation of the project. Some modules and files will be added as implementation progresses.*

## 8. Installation and Setup

### Prerequisites

- Python 3.10 or later (subject to dependency compatibility).
- Visual Studio Code.
- Jupyter Notebook extension in VS Code.
- Git.

### Step 1: Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd Hybrid-Movie-Recommendation-System
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up the Dataset

Download the HetRec 2011 – MovieLens 2K dataset from:

https://grouplens.org/datasets/hetrec-2011/

Extract the original `.dat` files into:

```text
dataset/raw/
```

The raw dataset is not committed to the repository. Each team member must obtain their own local copy.

### Step 5: Run Common Preprocessing

Open the following notebook in VS Code:

```text
notebooks/00_common_preprocessing.ipynb
```

Execute its cells to load, inspect, preprocess, and prepare the shared datasets.

### Step 6: Run Individual Modules

Open the notebook corresponding to your assigned research area and execute the implementation cells.

## 9. Experimental Evaluation

The recommendation system will be evaluated using appropriate metrics for rating prediction and recommendation ranking.

| Metric | Purpose |
|---|---|
| RMSE | Measures rating prediction error |
| MAE | Measures average absolute rating prediction error |
| Precision@K | Measures the proportion of relevant recommended items |
| Recall@K | Measures the proportion of relevant items successfully recommended |
| NDCG@K | Measures the ranking quality of recommendations |
| Catalog Coverage | Measures the diversity of movies recommended |
| Runtime | Measures computational efficiency |

The evaluation protocol will be adapted to the requirements of each research challenge.

## 10. Team Collaboration and Git Workflow

Each member works on a separate Git branch to avoid conflicts in the shared codebase.

### Branch Allocation

| Branch | Assigned Module |
|---|---|
| `main` | Stable integrated project |
| `data-sparsity` | Data Sparsity |
| `user-cold-start` | User Cold Start |
| `item-cold-start` | Item Cold Start |
| `system-cold-start` | System Cold Start |

### Workflow

1. Clone the shared repository.
2. Pull the latest changes from `main`.
3. Switch to your assigned branch.
4. Implement your assigned module in its respective notebook and source files.
5. Commit and push your changes.
6. Create a Pull Request to merge your work into `main`.
7. Review and test the integrated implementation before merging.

### Useful Git Commands

```bash
# Get the latest repository changes
git pull origin main

# Create and switch to your assigned branch
git checkout -b system-cold-start

# Stage changes
git add .

# Commit changes
git commit -m "Implement system cold start module"

# Push your branch
git push -u origin system-cold-start
```

Replace the branch name with your assigned branch when working on another module.

**Important Collaboration Guidelines:**

- Do not directly modify another member's notebook without coordination.
- Avoid committing large raw datasets and generated model files.
- Keep common preprocessing consistent across all modules.
- Use the shared processed datasets rather than independently modifying raw data.
- Document experimental settings and evaluation results.
- Merge changes into `main` through Pull Requests.

## 11. Expected Outcome

The expected outcome is a hybrid movie recommendation framework capable of generating recommendations under different interaction-data conditions.

The project aims to demonstrate how content-based and collaborative approaches can complement each other, particularly when a recommendation platform experiences limited user activity, new users, new movies, or a complete absence of historical interactions.

---

## 12. References

1. Cantador, I., Brusilovsky, P., & Kuflik, T. (2011). *2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011).* ACM.
2. GroupLens Research. *HetRec 2011 – MovieLens 2K Dataset.* https://grouplens.org/datasets/hetrec-2011/
3. Ricci, F., Rokach, L., & Shapira, B. (Eds.). *Recommender Systems Handbook.* Springer.

---

**Project Type:** Academic Research and Implementation Project  
**Domain:** Machine Learning | Recommendation Systems  
**Dataset:** HetRec 2011 – MovieLens 2K  
**Language:** Python
