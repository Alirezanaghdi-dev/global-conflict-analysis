"""
Global Conflict Data Analysis
==============================
End-to-end pipeline for exploring global conflict/violence data:
  1. Data loading and cleaning
  2. Exploratory Data Analysis (distributions, trends, correlations)
  3. Regional clustering (KMeans)
  4. Outlier removal and dimensionality reduction (PCA)
  5. Temporal risk analysis
  6. Machine learning model comparison
  7. Hyperparameter tuning of the best model (CatBoost)
  8. Saving ("deploying") the best trained model for reuse
"""

import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    silhouette_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

DATA_FILE = "../data/conflict_data.csv"
MODEL_OUTPUT_PATH = "../model/best_model.pkl"

# Columns that hold death counts for each conflict type, plus the year column.
# Kept together because the original analysis groups/aggregates them as one set.
DEATH_COLUMNS = [
    "Conflict_Year",
    "One_Sided_Violence_Deaths",
    "Non_State_Actor_Violence",
    "Intrastate_Conflict_Deaths",
    "Interstate_War_Deaths",
]

# Same as DEATH_COLUMNS but without the year column, for aggregations where
# summing the year itself would not make sense (e.g. regional totals).
DEATH_ONLY_COLUMNS = [col for col in DEATH_COLUMNS if col != "Conflict_Year"]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the conflict dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return data
    except FileNotFoundError:
        print(f"Could not find the file at {file_path}. Please verify the path.")
        raise
    except Exception as exc:
        print(f"An error occurred during dataset loading: {exc}")
        raise


# ---------------------------------------------------------------------------
# 2. Data preparation
# ---------------------------------------------------------------------------
def inspect_dataset(data: pd.DataFrame) -> None:
    """Print a quick overview of the dataset (columns, types, stats, missing values)."""
    print("Overview of the first 5 rows:")
    print(data.head().to_string())

    print("\nDataset columns and their data types:")
    print(data.info())

    print("\nDescriptive statistics:")
    print(data.describe().to_string())

    print("\nMissing data check:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\nUnique values summary for categorical columns:")
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        unique_values = data[col].unique()
        print(f"- {col}: {len(unique_values)} unique values")
        print(f"  values: {unique_values}")


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns to clearer, analysis-friendly names."""
    new_column_names = {
        "Entity": "Region",
        "Year": "Conflict_Year",
        "One-sided violence": "One_Sided_Violence_Deaths",
        "Non-state": "Non_State_Actor_Violence",
        "Intrastate": "Intrastate_Conflict_Deaths",
        "Extrasystemic": "Colonial_Conflict_Deaths",
        "Interstate": "Interstate_War_Deaths",
    }
    return data.rename(columns=new_column_names)


def standardize_text_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all categorical (text) columns for consistency."""
    categorical_columns = data.select_dtypes(include=["object"]).columns
    data[categorical_columns] = data[categorical_columns].apply(
        lambda col: col.str.lower() if col.dtype == "object" else col
    )
    return data


def check_numeric_ranges(data: pd.DataFrame, numeric_columns: list) -> None:
    """Print min/max ranges for numeric columns to spot anomalies."""
    range_summary = data[numeric_columns].agg(["min", "max"]).transpose()
    print("Checking numerical columns ranges for anomalies:")
    for col, row in range_summary.iterrows():
        print(f"- {col}: Min = {row['min']}, Max = {row['max']}")


def fix_negative_values(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Replace any negative values in the given columns with 0."""
    for col in columns:
        negative_mask = data[col] < 0
        if negative_mask.any():
            print(f"Negative values found in {col}. Replacing them with 0.")
            data.loc[negative_mask, col] = 0
        else:
            print(f"No negative values found in {col}.")
    return data


def prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline: rename, standardize, drop unused columns, fix negatives."""
    data = rename_columns(data)
    data = standardize_text_columns(data)

    # 'Colonial_Conflict_Deaths' is dropped because all its values are 0 (no useful signal)
    data = data.drop(columns=["Colonial_Conflict_Deaths"])

    data = fix_negative_values(data, DEATH_COLUMNS)
    return data


# ---------------------------------------------------------------------------
# 3. Level 1 - Foundational EDA
# ---------------------------------------------------------------------------
def plot_death_distributions(data: pd.DataFrame, columns: list) -> None:
    """Plot a histogram (with KDE) for each death-count column."""
    for column in columns:
        plt.figure(figsize=(9, 5))
        sns.histplot(data=data, x=column, kde=True, bins=20, color="green", alpha=0.6)
        plt.title(f"Histogram of {column.replace('_', ' ').title()} Distribution", fontsize=14)
        plt.xlabel("Number of Deaths", fontsize=11)
        plt.ylabel("Count", fontsize=11)
        plt.grid(visible=True, linestyle="-", alpha=0.3)
        plt.tight_layout()
        plt.show()


def summarize_annual_deaths(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Aggregate total deaths per year for each conflict type."""
    death_only_columns = [col for col in columns if col != "Conflict_Year"]
    annual_deaths = data.groupby("Conflict_Year")[death_only_columns].sum().reset_index()
    print(annual_deaths.to_string())
    return annual_deaths


def plot_annual_trends(annual_deaths: pd.DataFrame, columns: list) -> None:
    """Plot yearly trends for each conflict type as line charts."""
    plt.figure(figsize=(11, 7))
    for column in columns:
        if column == "Conflict_Year":
            continue
        plt.plot(
            annual_deaths["Conflict_Year"],
            annual_deaths[column],
            marker="o",
            label=column.replace("_", " ").title(),
        )

    plt.title("Annual Death Trends by Conflict Type", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Deaths", fontsize=12)
    plt.legend(title="Conflict Type")
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Level 2 - Intermediate EDA: relationships
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(data: pd.DataFrame, columns: list) -> None:
    """Plot a correlation heatmap for the death-count columns."""
    corr_matrix = data[columns].corr()
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    heatmap.set_title("Heatmap of Death Count Correlations")
    plt.show()


def plot_top_regions_per_conflict_type(data: pd.DataFrame, columns: list, top_n: int = 5) -> None:
    """For each conflict type, show and plot the regions with the highest total deaths."""
    filtered_data = data[data["Region"] != "world"]

    for column in columns:
        if column == "Conflict_Year":
            continue

        top_regions = (
            filtered_data.groupby("Region")[column].sum().sort_values(ascending=False).head(top_n)
        )

        print(f"Top {top_n} regions for {column.replace('_', ' ').title()}:")
        print(top_regions)

        plt.figure(figsize=(10, 6))
        top_regions.plot(kind="barh", color="purple", alpha=0.7)
        plt.title(f"Top {top_n} Regions with Maximum {column.replace('_', ' ').title()}")
        plt.xlabel("Total Deaths")
        plt.ylabel("Region")
        plt.grid(True)
        plt.show()


def add_total_deaths_column(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Add a 'total_deaths' column summing all conflict-type death counts per row."""
    data["total_deaths"] = data[columns].sum(axis=1)
    return data


def show_top_events_by_total_deaths(data: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Print and return the top N rows (events) with the highest total death count."""
    top_events = data.sort_values(by="total_deaths", ascending=False).head(top_n)
    print(f"Top {top_n} events with the highest total death count:")
    print(top_events.to_string())
    return top_events


def plot_stacked_deaths_by_year(data: pd.DataFrame, columns: list) -> None:
    """Plot a stacked bar chart of cumulative conflict-related deaths over the years."""
    death_only_columns = [col for col in columns if col != "Conflict_Year"]
    yearly_data = data.groupby("Conflict_Year")[death_only_columns].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    yearly_data.plot(x="Conflict_Year", kind="bar", stacked=True, colormap="viridis", ax=ax)

    ax.set_title("Cumulative Deaths by Conflict Type Over Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Deaths")
    ax.legend(title="Type of Conflict", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Level 3 - Advanced EDA: visual trends and regional comparison
# ---------------------------------------------------------------------------
def plot_bubble_chart_over_time(data: pd.DataFrame, columns: list) -> None:
    """Visualize total deaths over time per region using an interactive bubble chart."""
    fig = px.scatter(
        data,
        x="Conflict_Year",
        y="total_deaths",
        size="total_deaths",
        color="Region",
        hover_data=columns,
        title="Total Deaths Over Time by Region",
        labels={"total_deaths": "Total Deaths", "Region": "Geographical Region"},
    )
    fig.update_layout(width=1000, height=600)
    fig.show()


def plot_regional_trend_comparison(data: pd.DataFrame, regions: list) -> None:
    """Plot annual death trends for a chosen set of regions."""
    region_data = data[data["Region"].isin(regions)]

    if region_data.empty:
        print("No data available for the specified regions.")
        return

    plt.figure(figsize=(10, 6))
    for region in regions:
        annual_trend = region_data[region_data["Region"] == region].groupby("Conflict_Year")["total_deaths"].sum()

        plt.plot(
            annual_trend.index,
            annual_trend.values,
            marker="o",
            linestyle="--",
            linewidth=2,
            label=f"Deaths in {region.capitalize()}",
        )

    plt.title("Annual Death Trends in Selected Regions")
    plt.xlabel("Year")
    plt.ylabel("Total Death Count")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------
def find_optimal_cluster_count(data: pd.DataFrame, columns: list, exclude_region: str = "world") -> pd.DataFrame:
    """Aggregate death counts by region, then evaluate cluster counts via the elbow method and silhouette score."""
    clustering_data = data[data["Region"] != exclude_region].groupby("Region")[columns].sum()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    max_clusters = min(10, len(clustering_data) - 1)
    wcss_scores = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        wcss_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(range(2, max_clusters + 1), wcss_scores, marker="o", linestyle="--", color="b")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("WCSS", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker="o", linestyle="--", color="r")
    ax2.set_ylabel("Silhouette Score", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    plt.title("Elbow Method and Silhouette Score for Optimal Number of Clusters")
    plt.grid(True)
    plt.show()

    return clustering_data


def cluster_regions(data: pd.DataFrame, columns: list, n_clusters: int = 4, exclude_region: str = "world") -> tuple:
    """Cluster regions by aggregated death counts using KMeans, returning region-level and merged datasets."""
    clustering_data = data[data["Region"] != exclude_region].groupby("Region")[columns].sum()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clustering_data["Cluster"] = kmeans.fit_predict(scaled_data)
    clustering_data = clustering_data.reset_index()

    merged_data = data.merge(clustering_data[["Region", "Cluster"]], on="Region")
    return clustering_data, merged_data


def plot_cluster_heatmap(clustering_data: pd.DataFrame) -> None:
    """Plot a heatmap of aggregated death counts per region, grouped by cluster."""
    indexed_data = clustering_data.set_index("Region")

    plt.figure(figsize=(8, 6))
    sns.heatmap(indexed_data.drop(columns="Cluster"), annot=True, cmap="Blues", fmt=".0f")
    plt.title("Heatmap of Regional Conflict Clusters")
    plt.xlabel("Conflict Indicators")
    plt.ylabel("Regions")
    plt.xticks(rotation=45)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Statistical analysis
# ---------------------------------------------------------------------------
def summarize_statistics(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Compute descriptive statistics plus skewness and kurtosis for each column."""
    summary = data[columns].describe().T
    summary["Skewness"] = data[columns].skew()
    summary["Kurtosis"] = data[columns].kurt()
    print("Comprehensive statistical overview:")
    print(summary.to_string())
    return summary


def test_normality(data: pd.DataFrame, columns: list) -> None:
    """Run the Shapiro-Wilk normality test on each column."""
    print("Running normality test:")
    for column in columns:
        stat, p_value = stats.shapiro(data[column])
        result = "Data is normally distributed." if p_value > 0.05 else "Data is not normally distributed."
        print(f"{column}: Shapiro-Wilk test p-value = {p_value:.4f}. {result}")
        time.sleep(1)


def remove_outliers_zscore(data: pd.DataFrame, columns: list, z_threshold: float = 4) -> pd.DataFrame:
    """Remove rows where any column has a Z-score above the given threshold."""
    z_scores = np.abs(stats.zscore(data[columns]))
    outlier_indices = np.where(z_scores > z_threshold)
    print(f"Number of outliers detected with Z-score > {z_threshold}: {len(outlier_indices[0])}")

    cleaned_data = data[(z_scores < z_threshold).all(axis=1)].copy()

    if cleaned_data.empty:
        print("All data points were removed. Consider adjusting the Z-score threshold.")
    else:
        print(f"Outliers removed. Remaining rows: {cleaned_data.shape[0]}")

    return cleaned_data


# ---------------------------------------------------------------------------
# 7. Pattern discovery and dimensionality reduction
# ---------------------------------------------------------------------------
def cluster_cleaned_data(data: pd.DataFrame, columns: list, n_clusters: int = 4) -> pd.DataFrame:
    """Cluster rows (events) based on standardized death counts after outlier removal."""
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(data[columns])

    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    data["Clusters"] = kmeans_model.fit_predict(standardized_features)
    return data


def plot_cluster_scatter(data: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Plot a scatter chart comparing two death-count columns, colored by cluster."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster_label in sorted(data["Clusters"].unique()):
        cluster_data = data[data["Clusters"] == cluster_label]
        ax.scatter(cluster_data[x_col], cluster_data[y_col], label=f"Cluster {cluster_label}")

    ax.set_title("Clusters Based on Conflict Characteristics")
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_pca_projection(data: pd.DataFrame, feature_columns: list) -> None:
    """Reduce conflict death features to 2 dimensions with PCA and plot the clusters."""
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(data[feature_columns])

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(standardized_features)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        pca_components[:, 0],
        pca_components[:, 1],
        c=data["Clusters"],
        cmap="plasma",
        edgecolors="black",
        s=60,
        alpha=0.8,
    )

    ax.set_title("PCA Projection of Conflict Data", fontsize=14, fontweight="bold")
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)

    cbar = fig.colorbar(scatter)
    cbar.set_label("Clusters")

    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 8. Temporal and spatial risk analysis
# ---------------------------------------------------------------------------
def compute_risk_score_over_time(data: pd.DataFrame, columns: list) -> pd.Series:
    """Compute a yearly risk score as the sum of all death counts."""
    yearly_totals = data.groupby("Conflict_Year")[columns].sum()
    risk_score = yearly_totals.sum(axis=1)
    return risk_score


def plot_risk_score_with_peaks(risk_score: pd.Series, top_n: int = 3) -> None:
    """Plot the risk score over time and annotate the top N peak years."""
    top_years = risk_score.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(risk_score.index, risk_score.values, marker="o", color="purple", label="Risk Score")

    for year, value in top_years.items():
        ax.annotate(
            f"{year}: {int(value):,}",
            xy=(year, value),
            xytext=(year + 0.5, value + 0.5),
            textcoords="data",
            color="red",
            fontsize=10,
            arrowprops=dict(facecolor="red", arrowstyle="->"),
        )

    peak_year = top_years.index[0]
    ax.axvline(x=peak_year, color="blue", linestyle="--", label=f"Peak Year: {peak_year}")

    ax.set_title("Risk Score Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Aggregated Conflict Deaths (Risk Score)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def compute_yearly_growth_rates(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Compute year-on-year percentage growth for each conflict type, cleaning inf/NaN values."""
    yearly_totals = data.groupby("Conflict_Year")[columns].sum()
    growth_rates = yearly_totals.pct_change()
    growth_rates = growth_rates.replace([np.inf, -np.inf], 0).fillna(0)
    return growth_rates


def plot_growth_rates(growth_rates: pd.DataFrame) -> None:
    """Plot year-on-year growth rates for each conflict type."""
    plt.figure(figsize=(10, 6))
    for conflict_type in growth_rates.columns:
        plt.plot(
            growth_rates.index,
            growth_rates[conflict_type],
            marker="o",
            label=conflict_type.replace("_", " ").title(),
        )

    plt.title("Year-on-Year Growth Rate for Conflict Types")
    plt.xlabel("Year")
    plt.ylabel("Growth Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_cumulative_deaths_by_region(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Compute the cumulative sum of deaths per region across years."""
    region_yearly_sum = data.groupby(["Conflict_Year", "Region"], as_index=False)[columns].sum()

    for col in columns:
        region_yearly_sum[col] = region_yearly_sum.groupby("Region")[col].cumsum()

    region_yearly_sum["Total_Cumulative"] = region_yearly_sum[columns].sum(axis=1)
    return region_yearly_sum


def plot_cumulative_deaths_top_regions(data: pd.DataFrame, columns: list, top_n: int = 3) -> None:
    """Plot cumulative death trends over time for the top N regions by total deaths."""
    cumulative_data = compute_cumulative_deaths_by_region(data, columns)
    top_regions = data.groupby("Region")[columns].sum().sum(axis=1).nlargest(top_n).index.tolist()

    plt.figure(figsize=(10, 6))
    for region in top_regions:
        region_df = cumulative_data[cumulative_data["Region"] == region]
        plt.plot(region_df["Conflict_Year"], region_df["Total_Cumulative"], marker="o", label=region.title())

    plt.title("Cumulative Impact of Conflict by Region Over Time")
    plt.xlabel("Year")
    plt.ylabel("Cumulative Deaths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 9. Feature engineering and train/test split for ML
# ---------------------------------------------------------------------------
def build_ml_dataset(data: pd.DataFrame, death_columns: list) -> pd.DataFrame:
    """One-hot encode the Region column and drop columns not used as ML features."""
    data = add_total_deaths_column(data, death_columns)

    ml_data = pd.get_dummies(data, columns=["Region"], drop_first=True)
    ml_data = ml_data.drop(columns=["Conflict_Year", "Clusters"])
    return ml_data


def split_and_scale_data(ml_data: pd.DataFrame, target_column: str = "total_deaths"):
    """Split features/target into train/test sets and scale the features."""
    X = ml_data.drop(columns=[target_column])
    y = ml_data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def compare_regression_models(X_train_scaled, X_test_scaled, y_train, y_test) -> list:
    """Train and evaluate a set of regression models, returning their performance metrics."""
    model_list = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
        ("XGBoost", XGBRFRegressor(random_state=42, verbosity=0)),
        ("Support Vector Regression", SVR()),
        ("K-Nearest Neighbors", KNeighborsRegressor()),
        ("CatBoost", CatBoostRegressor(verbose=0, random_state=42)),
        ("LightGBM", LGBMRegressor(random_state=42)),
    ]

    model_performance = []

    for name, model in model_list:
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae_val = mean_absolute_error(y_test, y_pred)
        r2_val = r2_score(y_test, y_pred)

        model_performance.append(
            {
                "Model": name,
                "MAE": round(mae_val, 2),
                "R2_Score": round(r2_val, 4),
            }
        )
        time.sleep(1)

    return model_performance


# ---------------------------------------------------------------------------
# 10. GradientBoostingRegressor hyperparameter tuning, evaluation, and deployment
# ---------------------------------------------------------------------------
def tune_gradientboostingregressor_model(X_train_scaled, y_train) -> RandomizedSearchCV:
    """Run a randomized hyperparameter search for a GradientBoosting Regressor."""
    base_model = GradientBoostingRegressor(random_state=42)

    param_distributions = {
        "n_estimators": [100, 200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 6, 8, 10],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 3, 5, 10],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", None],
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        n_jobs=-1,
        scoring="r2",
        cv=10,
        verbose=1,
        random_state=42,
    )

    random_search.fit(X_train_scaled, y_train)
    return random_search


def evaluate_model(model, X_test_scaled, y_test) -> None:
    """Print R2 and MAE scores for the given model on the test set."""
    y_pred = model.predict(X_test_scaled)
    print(f"R2 Score on Test Set: {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Absolute Error on Test Set: {mean_absolute_error(y_test, y_pred):.2f}")


def plot_feature_importance(model, feature_names) -> None:
    """Plot feature importance scores for a trained GradientBoosting Regressor model."""
    feature_importance = model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.title("Feature Importance (GradientBoosting Regressor)")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


def drop_region_columns(X_train_scaled, X_test_scaled, feature_names) -> tuple:
    """Convert scaled arrays back to DataFrames and drop one-hot encoded Region columns."""
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    region_columns = [col for col in X_train_scaled_df.columns if col.startswith("Region_")]
    X_train_reduced = X_train_scaled_df.drop(columns=region_columns)
    X_test_reduced = X_test_scaled_df.drop(columns=region_columns)

    return X_train_reduced, X_test_reduced


def deploy_best_model(model, output_path: str) -> None:
    """Persist the trained model to disk so it can be loaded and reused later (deployment artifact)."""
    joblib.dump(model, output_path)
    print(f"Best model deployed and saved to '{output_path}'.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Load data
    conflict_data = load_dataset(DATA_FILE)
    inspect_dataset(conflict_data)

    # 2. Prepare / clean data
    conflict_data = prepare_dataset(conflict_data)
    numeric_columns = conflict_data.select_dtypes(include=["int", "float"]).columns
    check_numeric_ranges(conflict_data, numeric_columns)

    # 3. Foundational EDA
    plot_death_distributions(conflict_data, DEATH_COLUMNS)
    annual_deaths = summarize_annual_deaths(conflict_data, DEATH_COLUMNS)
    plot_annual_trends(annual_deaths, DEATH_COLUMNS)

    # 4. Intermediate EDA: relationships
    plot_correlation_heatmap(conflict_data, DEATH_COLUMNS)
    plot_top_regions_per_conflict_type(conflict_data, DEATH_COLUMNS)

    conflict_data = add_total_deaths_column(conflict_data, DEATH_COLUMNS)
    show_top_events_by_total_deaths(conflict_data)
    plot_stacked_deaths_by_year(conflict_data, DEATH_COLUMNS)

    # 5. Advanced EDA: trends and regional comparison
    plot_bubble_chart_over_time(conflict_data, DEATH_COLUMNS)
    plot_regional_trend_comparison(conflict_data, ["africa", "asia and oceania"])

    # Regional clustering
    find_optimal_cluster_count(conflict_data, DEATH_COLUMNS)
    clustering_data, conflict_data = cluster_regions(conflict_data, DEATH_COLUMNS, n_clusters=4)
    plot_cluster_heatmap(clustering_data)

    # 6. Statistical analysis
    summarize_statistics(conflict_data, DEATH_COLUMNS)
    test_normality(conflict_data, DEATH_COLUMNS)

    cleaned_conflict_data = remove_outliers_zscore(conflict_data, DEATH_COLUMNS, z_threshold=4)

    # 7. Pattern discovery after outlier removal
    cleaned_conflict_data = cluster_cleaned_data(cleaned_conflict_data, DEATH_COLUMNS, n_clusters=4)
    cleaned_conflict_data = cleaned_conflict_data.drop(columns=["Cluster"])

    plot_cluster_scatter(cleaned_conflict_data, "One_Sided_Violence_Deaths", "Intrastate_Conflict_Deaths")
    plot_cluster_scatter(cleaned_conflict_data, "Non_State_Actor_Violence", "Intrastate_Conflict_Deaths")

    pca_feature_columns = [
        "One_Sided_Violence_Deaths",
        "Non_State_Actor_Violence",
        "Intrastate_Conflict_Deaths",
        "Interstate_War_Deaths",
    ]
    plot_pca_projection(cleaned_conflict_data, pca_feature_columns)

    # 8. Temporal and spatial risk analysis
    risk_score = compute_risk_score_over_time(conflict_data, DEATH_COLUMNS)
    plot_risk_score_with_peaks(risk_score, top_n=3)

    growth_rates = compute_yearly_growth_rates(conflict_data, DEATH_COLUMNS)
    plot_growth_rates(growth_rates)

    plot_cumulative_deaths_top_regions(conflict_data, DEATH_COLUMNS, top_n=3)

    # 9. Feature engineering and model comparison
    ml_data = build_ml_dataset(cleaned_conflict_data, DEATH_COLUMNS)
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale_data(ml_data)

    model_performance = compare_regression_models(X_train_scaled, X_test_scaled, y_train, y_test)
    print("Model performance comparison:")
    print(pd.DataFrame(model_performance).to_string())

    # 10. GradientBoosting Regressor tuning and deployment
    random_search = tune_gradientboostingregressor_model(X_train_scaled, y_train)
    best_model = random_search.best_estimator_

    print("Best parameters found:", random_search.best_params_)
    evaluate_model(best_model, X_test_scaled, y_test)
    plot_feature_importance(best_model, X_train.columns)

    # Drop one-hot encoded Region columns from the scaled feature set (kept for reference/inspection)
    X_train_reduced, X_test_reduced = drop_region_columns(X_train_scaled, X_test_scaled, X_train.columns)
    print(X_train_reduced.head())
    print(X_test_reduced.head())

    # Save the best-performing model as the deployment artifact
    deploy_best_model(best_model, MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()