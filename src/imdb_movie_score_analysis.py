from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from matplotlib.ticker import FuncFormatter

DATA_PATH = PROJECT_ROOT / "data" / "imdb_movies.csv"
RANDOM_STATE = 42
KMEANS_N_CLUSTERS = 4

NUMERIC_MODEL_FEATURES = [
    "log_budget",
    "log_revenue",
    "log_profit_shifted",
    "roi_capped",
    "release_year",
    "is_english",
]

CATEGORICAL_MODEL_FEATURES = []

def ensure_output_dirs() -> None:
    """Create output folders used by the project."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    """Normalize text/categorical values and replace missing values."""
    if pd.isna(value):
        return "Unknown"

    text = str(value).replace("\xa0", " ").strip()
    return text if text else "Unknown"


def write_data_quality_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Save a small summary of dataset cleaning and data quality checks."""
    budget_col = "budget_x" if "budget_x" in raw_df.columns else "budget"
    revenue_col = "revenue"

    raw_budget = pd.to_numeric(raw_df[budget_col], errors="coerce")
    raw_revenue = pd.to_numeric(raw_df[revenue_col], errors="coerce")
    raw_score = pd.to_numeric(raw_df["score"], errors="coerce")

    summary = {
        "original_rows": len(raw_df),
        "cleaned_rows": len(cleaned_df),
        "rows_removed": len(raw_df) - len(cleaned_df),
        "percent_rows_removed": round(
            (len(raw_df) - len(cleaned_df)) / len(raw_df), 4
        ),
        "missing_genre_count": int(raw_df["genre"].isna().sum()),
        "missing_genre_rate": round(float(raw_df["genre"].isna().mean()), 4),
        "missing_crew_count": int(raw_df["crew"].isna().sum())
        if "crew" in raw_df.columns
        else None,
        "missing_crew_rate": round(float(raw_df["crew"].isna().mean()), 4)
        if "crew" in raw_df.columns
        else None,
        "zero_budget_count": int((raw_budget == 0).sum()),
        "zero_budget_rate": round(float((raw_budget == 0).mean()), 4),
        "zero_revenue_count": int((raw_revenue == 0).sum()),
        "zero_revenue_rate": round(float((raw_revenue == 0).mean()), 4),
        "score_min": float(raw_score.min()),
        "score_max": float(raw_score.max()),
        "score_mean": float(raw_score.mean()),
        "score_median": float(raw_score.median()),
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "data_quality_summary.csv", index=False
    )
    
def group_rare_categories(
    df: pd.DataFrame,
    column: str,
    min_count: int = 50,
) -> pd.Series:
    """Group infrequent category values into Other."""
    counts = df[column].value_counts()
    common_values = counts[counts >= min_count].index
    return df[column].where(df[column].isin(common_values), "Other")


def get_genre_features(df: pd.DataFrame) -> list[str]:
    """Return all multi-hot encoded genre feature columns."""
    return [col for col in df.columns if col.startswith("genre_")]


def get_model_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature lists for the main models."""
    numeric_features = NUMERIC_MODEL_FEATURES + get_genre_features(df)
    categorical_features = CATEGORICAL_MODEL_FEATURES
    return numeric_features, categorical_features


def load_and_clean_data() -> pd.DataFrame:
    """Load the IMDb dataset, clean fields, and create engineered features."""
    raw_df = pd.read_csv(DATA_PATH)
    df = raw_df.copy()

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
    df["genre"] = df["genre"].str.replace(", ", ",", regex=False)
    df["main_genre"] = df["genre"].str.split(",").str[0].apply(normalize_text)

    genre_dummies = (
        df["genre"]
        .str.get_dummies(sep=",")
        .rename(columns=lambda c: "genre_" + c.strip().replace(" ", "_"))
    )
    df = pd.concat([df, genre_dummies], axis=1)

    df["language"] = df["language"].apply(normalize_text)
    df["is_english"] = (df["language"].str.lower() == "english").astype(int)
    df["country"] = df["country"].apply(normalize_text)
    df["status"] = df["status"].apply(normalize_text)

    numeric_cols = ["score", "budget", "revenue"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["profit"] = df["revenue"] - df["budget"]

    df["roi"] = np.where(
        df["budget"] > 0,
        df["profit"] / df["budget"],
        np.nan,
    )

    roi_bounds = df["roi"].replace([np.inf, -np.inf], np.nan).quantile([0.01, 0.99])
    df["roi_capped"] = df["roi"].clip(
        lower=roi_bounds.loc[0.01],
        upper=roi_bounds.loc[0.99],
    )

    df["log_budget"] = np.log1p(df["budget"].clip(lower=0))
    df["log_revenue"] = np.log1p(df["revenue"].clip(lower=0))

    # Profit can be negative, so it is shifted before log-transforming.
    df["log_profit_shifted"] = np.log1p(
        (df["profit"] - df["profit"].min()).clip(lower=0)
    )
    
    cleaned = df[df["status"].eq("Released")].copy()
    cleaned = cleaned.dropna(subset=["score", "budget", "revenue"])
    cleaned = cleaned[
        (cleaned["score"] > 0)
        & (cleaned["budget"] >= 0)
        & (cleaned["revenue"] >= 0)
    ]

    # Group rare categories after filtering 
    for col in ["main_genre", "language", "country"]:
        cleaned[col] = group_rare_categories(cleaned, col, min_count=50)

    write_data_quality_summary(raw_df, cleaned)
    cleaned.to_csv(OUTPUT_DIR / "cleaned_movies.csv", index=False)
    return cleaned


def save_plot(filename: str) -> None:
    """Save and close the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()
    
def money_formatter(x, pos=None):
    """Format raw dollar values for plot axes."""
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1_000_000_000:
        return f"${x / 1_000_000_000:.1f}B"
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.0f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"

def log_money_formatter(x, pos=None):
    """Format log1p-dollar values as the original dollar scale."""
    raw_value = np.expm1(x)
    return money_formatter(raw_value, pos)



def make_eda_visualizations(df: pd.DataFrame) -> None:
    """Create exploratory plots for distributions, relationships, and categories."""
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
    plt.gca().xaxis.set_major_formatter(FuncFormatter(money_formatter))
    save_plot("hist_budget.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["revenue"], bins=40, color="#D17A22")
    plt.title("Distribution of Movie Revenue")
    plt.xlabel("Revenue")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(money_formatter))
    save_plot("hist_revenue.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["log_budget"], bins=40, kde=True, color="#7B9E89")
    plt.title("Distribution of Log Movie Budgets")
    plt.xlabel("Budget shown on log scale")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(log_money_formatter))
    save_plot("hist_log_budget.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["log_revenue"], bins=40, kde=True, color="#D17A22")
    plt.title("Distribution of Log Movie Revenue")
    plt.xlabel("Revenue shown on log scale")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(log_money_formatter))
    save_plot("hist_log_revenue.png")
    
    positive_budget = df[df["budget"] > 0].copy()
    positive_revenue = df[df["revenue"] > 0].copy()

    # Raw budget/revenue vs. score: useful for showing skew and motivating log transforms.
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=positive_budget, x="budget", y="score", alpha=0.35, edgecolor=None)
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(money_formatter))
    plt.title("Raw Budget vs. Score")
    plt.xlabel("Budget")
    plt.ylabel("Score")
    save_plot("scatter_budget_score.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=positive_revenue, x="revenue", y="score", alpha=0.35, edgecolor=None)
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(money_formatter))
    plt.title("Raw Revenue vs. Score")
    plt.xlabel("Revenue")
    plt.ylabel("Score")
    save_plot("scatter_revenue_score.png")

    # log budget/revenue vs. score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=positive_budget, x="log_budget", y="score", alpha=0.35, edgecolor=None)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(log_money_formatter))
    plt.title("Log Budget vs. Score")
    plt.xlabel("Budget shown on log scale")
    plt.ylabel("Score")
    save_plot("scatter_log_budget_score.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=positive_revenue, x="log_revenue", y="score", alpha=0.35, edgecolor=None)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(log_money_formatter))
    plt.title("Log Revenue vs. Score")
    plt.xlabel("Revenue shown on log scale")
    plt.ylabel("Score")
    save_plot("scatter_log_revenue_score.png")

    for col, filename, title in [
        ("main_genre", "box_score_by_genre.png", "Score by Main Genre"),
        ("language", "box_score_by_language.png", "Score by Original Language"),
    ]:
        top_values = df[col].value_counts().head(10).index
        subset = df[df[col].isin(top_values)].copy()

        plt.figure(figsize=(10, 5))
        order = subset.groupby(col)["score"].median().sort_values(
            ascending=False
        ).index

        sns.boxplot(data=subset, x=col, y="score", order=order, color="#88BDBC")
        plt.title(title)
        plt.xlabel(col.replace("_", " ").title())
        plt.xticks(rotation=35, ha="right")
        save_plot(filename)

    # correlation heatmap for numeric variables
    numeric_corr_cols = [
        "score",
        "budget",
        "revenue",
        "profit",
        "roi_capped",
        "release_year",
        "is_english",
        "log_budget",
        "log_revenue",
        "log_profit_shifted",
    ]

    plt.figure(figsize=(9, 7))
    corr = df[numeric_corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap of Numeric Features")
    save_plot("correlation_heatmap.png")


def regression_metrics(
    y_true: pd.Series,
    preds: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    """Return standard regression metrics."""
    return {
        "model": model_name,
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae": float(mean_absolute_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
    }


def mean_score_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Baseline that always predicts the training-set mean score."""
    y = df["score"]

    y_train, y_test = train_test_split(
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_mean = y_train.mean()
    mean_prediction = np.full(shape=len(y_test), fill_value=train_mean)

    return regression_metrics(y_test, mean_prediction, "Mean score baseline")


def regression_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Simple linear regression baseline using only budget and revenue."""
    X = df[["budget", "revenue"]]
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
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
    """Linear regression with engineered numeric features and encoded categories."""
    numeric_features, categorical_features = get_model_feature_lists(df)

    X = df[numeric_features + categorical_features]
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
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


def save_regression_coefficients(model: Pipeline) -> None:
    """Save improved regression coefficients for interpretation."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = regressor.coef_

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    coef_df.to_csv(OUTPUT_DIR / "regression_coefficients.csv", index=False)
    coef_df.head(20).to_csv(
        OUTPUT_DIR / "top_regression_coefficients.csv",
        index=False,
    )


def add_score_category(
    df: pd.DataFrame,
    save_cutoffs: bool = False,
) -> pd.DataFrame:
    """Create three roughly balanced score categories using quantile binning."""
    data = df.copy()

    # qcut finds score thresholds that split the data into roughly equal groups.
    # We then use cut with those thresholds so we can also save the category ranges.
    _, bin_edges = pd.qcut(
        data["score"],
        q=3,
        retbins=True,
        duplicates="drop",
    )

    labels = ["low", "medium", "high"][: len(bin_edges) - 1]

    data["score_category"] = pd.cut(
        data["score"],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
    )

    data = data.dropna(subset=["score_category"])

    if save_cutoffs:
        cutoff_rows = []

        for i, label in enumerate(labels):
            cutoff_rows.append(
                {
                    "category": label,
                    "score_lower_bound": float(bin_edges[i]),
                    "score_upper_bound": float(bin_edges[i + 1]),
                }
            )

        pd.DataFrame(cutoff_rows).to_csv(
            OUTPUT_DIR / "score_category_cutoffs.csv",
            index=False,
        )

    return data

def classification_metrics(
    y_true: pd.Series,
    preds: np.ndarray,
    model_name: str,
    category_distribution: dict[str, float],
) -> dict[str, float]:
    """Return standard multi-class classification metrics."""
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision_macro": float(
            precision_score(y_true, preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, preds, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, preds, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, preds, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, preds, average="weighted", zero_division=0)
        ),
        "low_rate": float(category_distribution.get("low", 0)),
        "medium_rate": float(category_distribution.get("medium", 0)),
        "high_rate": float(category_distribution.get("high", 0)),
    }


def classification_majority_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Baseline that always predicts the most common score category."""
    data = add_score_category(df)
    y = data["score_category"].astype(str)

    y_train, y_test = train_test_split(
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    majority_class = y_train.mode()[0]
    preds = np.full(shape=len(y_test), fill_value=majority_class)
    category_distribution = y.value_counts(normalize=True).to_dict()

    return classification_metrics(
        y_test,
        preds,
        "Majority class baseline: score category",
        category_distribution,
    )


def classification_model(df: pd.DataFrame) -> dict[str, float]:
    """Logistic regression classification for low/medium/high score categories."""
    data = add_score_category(df, save_cutoffs=True)

    numeric_features, categorical_features = get_model_feature_lists(data)

    X = data[numeric_features + categorical_features]
    y = data["score_category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(numeric_features, categorical_features),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    category_distribution = y.value_counts(normalize=True).to_dict()

    labels = ["low", "medium", "high"]

    confusion = pd.crosstab(
        y_test,
        preds,
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    confusion.to_csv(OUTPUT_DIR / "score_category_confusion_matrix.csv")

    actual_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    correct_counts = pd.Series(np.diag(confusion), index=labels)

    error_analysis = pd.DataFrame(
        {
            "actual_count": actual_counts,
            "predicted_count": predicted_counts,
            "correct_predictions": correct_counts,
            "recall": correct_counts / actual_counts,
            "precision": correct_counts / predicted_counts.replace(0, np.nan),
        }
    ).round(4)

    error_analysis.to_csv(
        OUTPUT_DIR / "classification_error_analysis.csv",
        index_label="score_category",
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title("Score Category Classification Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("score_category_confusion_matrix.png")

    return classification_metrics(
        y_test,
        preds,
        "Logistic regression classification: score quantile category",
        category_distribution,
    )


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
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


def plot_kmeans_elbow(df: pd.DataFrame) -> None:
    """Generate elbow plot to support the chosen number of K-Means clusters."""
    numeric_features, categorical_features = get_model_feature_lists(df)
    features = numeric_features + categorical_features

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    X_processed = preprocessor.fit_transform(df[features])

    inertias = []
    k_values = range(2, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        kmeans.fit(X_processed)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertias, marker="o")
    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    save_plot("kmeans_elbow_method.png")
    
def plot_kmeans_silhouette(df: pd.DataFrame) -> None:
    """Generate silhouette score plot to support the chosen number of K-Means clusters."""
    numeric_features, categorical_features = get_model_feature_lists(df)
    features = numeric_features + categorical_features

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    X_processed = preprocessor.fit_transform(df[features])

    silhouette_scores = []
    k_values = range(2, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = kmeans.fit_predict(X_processed)
        silhouette_scores.append(silhouette_score(X_processed, labels))

    silhouette_df = pd.DataFrame(
        {
            "k": list(k_values),
            "silhouette_score": silhouette_scores,
        }
    )

    silhouette_df.to_csv(
        OUTPUT_DIR / "kmeans_silhouette_scores.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), silhouette_scores, marker="o")
    plt.title("K-Means Silhouette Scores")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    save_plot("kmeans_silhouette_scores.png")


def clustering_and_pca(df: pd.DataFrame) -> pd.DataFrame:
    """Run K-Means clustering and visualize clusters using PCA."""
    numeric_features, categorical_features = get_model_feature_lists(df)
    features = numeric_features + categorical_features

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    X_processed = preprocessor.fit_transform(df[features])

    kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_processed)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_processed)
    
    pca_variance = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    pca_variance.to_csv(
        OUTPUT_DIR / "pca_explained_variance.csv",
        index=False,
    )

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
        )
        .round(3)
        .reset_index()
    )

    profile.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)

    clustered[
        [
            "title",
            "score",
            "budget",
            "revenue",
            "profit",
            "main_genre",
            "language",
            "is_english",
            "country",
            "cluster",
        ]
    ].to_csv(OUTPUT_DIR / "movies_with_clusters.csv", index=False)

    return profile


def save_results(results: list[dict[str, float]]) -> None:
    """Save combined, regression-only, and classification-only result tables."""
    results_df = pd.DataFrame([dict(result) for result in results])

    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    with open(OUTPUT_DIR / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)

    regression_results = results_df[results_df["rmse"].notna()].copy()
    classification_results = results_df[results_df["accuracy"].notna()].copy()

    regression_results.to_csv(OUTPUT_DIR / "regression_results.csv", index=False)
    classification_results.to_csv(
        OUTPUT_DIR / "classification_results.csv",
        index=False,
    )

def write_summary(
    df: pd.DataFrame,
    results: list[dict[str, float]],
    cluster_profile: pd.DataFrame,
) -> None:
    """Write a short markdown summary of generated project outputs."""
    top_genres = (
        df.groupby("main_genre")
        .agg(
            movie_count=("title", "count"),
            avg_score=("score", "mean"),
        )
        .query("movie_count >= 50")
        .sort_values("avg_score", ascending=False)
        .head(8)
        .round(2)
    )

    lines = [
        "# Auto-Generated Project Summary",
        "",
        f"Cleaned dataset size: {len(df):,} released movies.",
        (
            f"Score range: {df['score'].min():.1f} to {df['score'].max():.1f}; "
            f"mean score: {df['score'].mean():.2f}."
        ),
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
            "Use the mean score baseline to show what performance looks like without using "
            "movie features. Use the baseline linear regression to discuss whether budget "
            "and revenue alone predict score. Use the improved regression to evaluate whether "
            "metadata adds predictive value. Use the majority-class baseline and logistic "
            "regression classifier to evaluate score-category classification. Use the cluster "
            "profiles to describe the types of movies that appear together based on financial "
            "and metadata features, then compare their average scores."
        ),
    ]

    (OUTPUT_DIR / "project_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    ensure_output_dirs()
    df = load_and_clean_data()

    print(f"Loaded and cleaned {len(df):,} movies.")

    print("Creating EDA visualizations...")
    make_eda_visualizations(df)

    print("Running mean score baseline...")
    mean_baseline = mean_score_baseline(df)

    print("Training baseline regression...")
    baseline = regression_baseline(df)

    print("Training improved regression...")
    improved, improved_model = improved_regression(df)

    print("Saving regression coefficients...")
    save_regression_coefficients(improved_model)

    print("Running classification majority baseline...")
    classification_baseline = classification_majority_baseline(df)

    print("Training classification model...")
    classification = classification_model(df)

    print("Creating K-Means elbow plot...")
    plot_kmeans_elbow(df)
    
    print("Creating K-Means silhouette plot...")
    plot_kmeans_silhouette(df)

    print("Running K-Means and PCA...")
    cluster_profile = clustering_and_pca(df)

    results = [
        mean_baseline,
        baseline,
        improved,
        classification_baseline,
        classification,
    ]

    save_results(results)
    write_summary(df, results, cluster_profile)

    print("Done. Outputs saved in:", OUTPUT_DIR)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()