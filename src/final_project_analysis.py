"""
CS439 Final Project

Research question:
Can movie metadata and financial information explain or predict IMDB movie scores,
and what types of movies tend to perform best?

Sections:
1. EDA and visualizations
2. Linear regression baseline
3. Improved linear regression with engineered features
4. Classification: high-rated vs. not high-rated
5. K-Means clustering + PCA visualization
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PKGS = PROJECT_ROOT / ".venv_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = PROJECT_ROOT / "data" / "imdb_movies.csv"
RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).replace("\xa0", " ").strip()
    return text if text else "Unknown"


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df = df.rename(
        columns={
            "names": "title",
            "date_x": "release_date",
            "budget_x": "budget",
            "orig_lang": "language",
        }
    )

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["genre"] = df["genre"].apply(normalize_text)
    df["main_genre"] = df["genre"].str.split(",").str[0].apply(normalize_text)
    df["language"] = df["language"].apply(normalize_text)
    df["country"] = df["country"].apply(normalize_text)
    df["status"] = df["status"].apply(normalize_text)

    numeric_cols = ["score", "budget", "revenue"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["profit"] = df["revenue"] - df["budget"]
    df["roi"] = np.where(df["budget"] > 0, df["profit"] / df["budget"], np.nan)
    roi_bounds = df["roi"].replace([np.inf, -np.inf], np.nan).quantile([0.01, 0.99])
    df["roi_capped"] = df["roi"].clip(lower=roi_bounds.loc[0.01], upper=roi_bounds.loc[0.99])
    df["log_budget"] = np.log1p(df["budget"].clip(lower=0))
    df["log_revenue"] = np.log1p(df["revenue"].clip(lower=0))
    df["log_profit_shifted"] = np.log1p((df["profit"] - df["profit"].min()).clip(lower=0))

    # Keep only completed movies with usable score and financial fields.
    cleaned = df[df["status"].eq("Released")].copy()
    cleaned = cleaned.dropna(subset=["score", "budget", "revenue"])
    cleaned = cleaned[(cleaned["score"] > 0) & (cleaned["budget"] >= 0) & (cleaned["revenue"] >= 0)]

    cleaned.to_csv(OUTPUT_DIR / "cleaned_movies.csv", index=False)
    return cleaned


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def make_eda_visualizations(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["score"], bins=30, kde=True, color="#376996")
    plt.title("Distribution of Movie Scores")
    plt.xlabel("Score")
    save_plot("hist_score.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["budget"], bins=40, color="#7B9E89")
    plt.title("Distribution of Movie Budgets")
    plt.xlabel("Budget")
    save_plot("hist_budget.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["revenue"], bins=40, color="#D17A22")
    plt.title("Distribution of Movie Revenue")
    plt.xlabel("Revenue")
    save_plot("hist_revenue.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="budget", y="score", alpha=0.35, edgecolor=None)
    plt.xscale("symlog")
    plt.title("Budget vs. Score")
    save_plot("scatter_budget_score.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="revenue", y="score", alpha=0.35, edgecolor=None)
    plt.xscale("symlog")
    plt.title("Revenue vs. Score")
    save_plot("scatter_revenue_score.png")

    for col, filename, title in [
        ("main_genre", "box_score_by_genre.png", "Score by Main Genre"),
        ("language", "box_score_by_language.png", "Score by Original Language"),
        ("country", "box_score_by_country.png", "Score by Country"),
    ]:
        top_values = df[col].value_counts().head(10).index
        subset = df[df[col].isin(top_values)].copy()
        plt.figure(figsize=(10, 5))
        order = subset.groupby(col)["score"].median().sort_values(ascending=False).index
        sns.boxplot(data=subset, x=col, y="score", order=order, color="#88BDBC")
        plt.title(title)
        plt.xlabel(col.replace("_", " ").title())
        plt.xticks(rotation=35, ha="right")
        save_plot(filename)


def regression_baseline(df: pd.DataFrame) -> dict[str, float]:
    X = df[["budget", "revenue"]]
    y = df["score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return regression_metrics(y_test, preds, "Baseline linear regression")


def improved_regression(df: pd.DataFrame) -> tuple[dict[str, float], Pipeline]:
    numeric_features = [
        "log_budget",
        "log_revenue",
        "profit",
        "roi_capped",
        "release_year",
    ]
    categorical_features = ["main_genre", "language", "country"]

    X = df[numeric_features + categorical_features]
    y = df["score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return regression_metrics(y_test, preds, "Improved linear regression"), model


def regression_metrics(y_true: pd.Series, preds: np.ndarray, model_name: str) -> dict[str, float]:
    return {
        "model": model_name,
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae": float(mean_absolute_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
    }


def classification_model(df: pd.DataFrame) -> dict[str, float]:
    high_score_threshold = 75
    data = df.copy()
    data["high_rated"] = (data["score"] >= high_score_threshold).astype(int)

    numeric_features = ["log_budget", "log_revenue", "profit", "roi_capped", "release_year"]
    categorical_features = ["main_genre", "language", "country"]
    X = data[numeric_features + categorical_features]
    y = data["high_rated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": f"Logistic regression classification score >= {high_score_threshold}",
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "positive_rate": float(y.mean()),
    }


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def clustering_and_pca(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [
        "score",
        "log_budget",
        "log_revenue",
        "profit",
        "roi_capped",
        "release_year",
    ]
    categorical_features = ["main_genre", "language", "country"]
    features = numeric_features + categorical_features

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    X_processed = preprocessor.fit_transform(df[features])

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_processed)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_processed)

    clustered = df.copy()
    clustered["cluster"] = labels
    clustered["pca_1"] = components[:, 0]
    clustered["pca_2"] = components[:, 1]

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=clustered,
        x="pca_1",
        y="pca_2",
        hue="cluster",
        palette="Set2",
        alpha=0.65,
        s=35,
    )
    plt.title("K-Means Movie Clusters Visualized with PCA")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    save_plot("pca_kmeans_clusters.png")

    profile = (
        clustered.groupby("cluster")
        .agg(
            movie_count=("title", "count"),
            avg_score=("score", "mean"),
            median_budget=("budget", "median"),
            median_revenue=("revenue", "median"),
            median_profit=("profit", "median"),
            median_roi=("roi", "median"),
            median_year=("release_year", "median"),
            top_genre=("main_genre", lambda s: s.value_counts().index[0]),
            top_language=("language", lambda s: s.value_counts().index[0]),
            top_country=("country", lambda s: s.value_counts().index[0]),
        )
        .round(3)
        .reset_index()
    )
    profile.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)
    clustered[["title", "score", "budget", "revenue", "main_genre", "language", "country", "cluster"]].to_csv(
        OUTPUT_DIR / "movies_with_clusters.csv", index=False
    )
    return profile


def save_results(results: list[dict[str, float]]) -> None:
    rows = []
    for result in results:
        row = dict(result)
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    with open(OUTPUT_DIR / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def write_summary(df: pd.DataFrame, results: list[dict[str, float]], cluster_profile: pd.DataFrame) -> None:
    top_genres = (
        df.groupby("main_genre")
        .agg(movie_count=("title", "count"), avg_score=("score", "mean"))
        .query("movie_count >= 50")
        .sort_values("avg_score", ascending=False)
        .head(8)
        .round(2)
    )

    lines = [
        "# Auto-Generated Project Summary",
        "",
        f"Cleaned dataset size: {len(df):,} released movies.",
        f"Score range: {df['score'].min():.1f} to {df['score'].max():.1f}; mean score: {df['score'].mean():.2f}.",
        "",
        "## Top Genres by Average Score",
        "```text",
        top_genres.to_string(),
        "```",
        "",
        "## Model Results",
        "```text",
        pd.DataFrame(results).to_string(index=False),
        "```",
        "",
        "## Cluster Profiles",
        "```text",
        cluster_profile.to_string(index=False),
        "```",
        "",
        "## Interpretation Starter",
        (
            "Use the baseline model to discuss whether budget and revenue alone predict score. "
            "Use the improved regression and classification results to evaluate whether metadata "
            "adds predictive value. Use the cluster profiles to describe the types of movies that "
            "perform best, such as genres or financial profiles associated with higher average scores."
        ),
    ]
    (OUTPUT_DIR / "project_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    df = load_and_clean_data()

    print(f"Loaded and cleaned {len(df):,} movies.")
    print("Creating EDA visualizations...")
    make_eda_visualizations(df)

    print("Training baseline regression...")
    baseline = regression_baseline(df)

    print("Training improved regression...")
    improved, _ = improved_regression(df)

    print("Training classification model...")
    classification = classification_model(df)

    print("Running K-Means and PCA...")
    cluster_profile = clustering_and_pca(df)

    results = [baseline, improved, classification]
    save_results(results)
    write_summary(df, results, cluster_profile)

    print("Done. Outputs saved in:", OUTPUT_DIR)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
