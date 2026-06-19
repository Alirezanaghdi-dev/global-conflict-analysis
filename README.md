# Global Conflict Data Analysis

An end-to-end exploratory data analysis (EDA) and machine learning pipeline for
global conflict and violence data, including regional clustering, statistical
testing, dimensionality reduction, and a tuned GradientBoostingRegressor model.

## What this project does

1. **Data loading & cleaning** — loads the raw conflict dataset, renames columns,
   standardizes text fields, drops unused columns, and fixes invalid (negative) values.
2. **Exploratory Data Analysis**
   - Distribution histograms for each conflict type
   - Annual death trend lines
   - Correlation heatmap between conflict types
   - Top regions per conflict type
   - Top single events by total deaths
   - Stacked bar charts of deaths over time
3. **Advanced visual analysis**
   - Interactive bubble chart of deaths over time (Plotly)
   - Region-to-region trend comparisons
4. **Clustering**
   - Elbow method + silhouette score to pick the optimal number of clusters
   - KMeans clustering of regions by conflict profile
   - Cluster heatmap
5. **Statistical analysis**
   - Descriptive statistics, skewness, kurtosis
   - Shapiro-Wilk normality testing
   - Z-score based outlier removal
6. **Dimensionality reduction**
   - PCA projection of conflict features, colored by cluster
7. **Temporal & spatial risk analysis**
   - Yearly aggregated "risk score" with peak-year annotations
   - Year-on-year growth rate per conflict type
   - Cumulative deaths over time for the top regions
8. **Machine learning**
   - Feature engineering (one-hot encoding, train/test split, scaling)
   - Comparison of 8 regression models (Linear Regression, Random Forest,
     Gradient Boosting, XGBoost, SVR, KNN, CatBoost, LightGBM)
   - Hyperparameter tuning of CatBoost via `RandomizedSearchCV`
   - Feature importance plot
9. **Deployment**
   - The best-performing tuned model is saved to `models/best_model.pkl` via
     `joblib`, ready to be loaded and reused for inference in another script
     or service.

## Project structure

```
.
├──src
 ├── conflict_analysis.py   # Main analysis & ML pipeline
├── requirements.txt       # Python dependencies
├── .gitignore
├──model
 ├── models/                # Saved ("deployed") trained model (generated at runtime)
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Place your dataset as `conflict_data.csv` in the project root. Expected
   raw columns: `Entity`, `Year`, `One-sided violence`, `Non-state`,
   `Intrastate`, `Extrasystemic`, `Interstate`.
2. Run the full pipeline:

```bash
python conflict_analysis.py
```

This will print EDA summaries to the console, display all charts, train and
compare regression models, tune the best one (GradientBoostingRegressor), and save it to
`models/best_model.pkl`.

## Loading the deployed model later

```python
import joblib

model = joblib.load("models/best_model.pkl")
predictions = model.predict(X_new_scaled)
```

## Notes

- Random seeds are fixed (`random_state=42`) throughout for reproducibility.
- The Z-score outlier threshold (`4`) is intentionally relaxed to avoid
  removing too much real conflict data.


[https://github.com/Alirezanaghdi-dev/global-conflict-analysis](https://github.com/Alirezanaghdi-dev/global-conflict-analysis)
