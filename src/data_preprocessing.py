import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# 1. Load Data
# -----------------------------
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


# -----------------------------
# 2. Reshape Data
# -----------------------------
def reshape_data(df: pd.DataFrame) -> pd.DataFrame:
    df_long = df.melt(
        id_vars=["Country Name", "Time"],
        var_name="Indicators",
        value_name="Value"
    )

    df_wide = df_long.pivot_table(
        index=["Country Name", "Time"],
        columns="Indicators",
        values="Value"
    ).reset_index()

    return df_wide


# -----------------------------
# 3. Prune Missing Data
# -----------------------------
def prune_missing(df: pd.DataFrame, col_thresh=0.5, row_thresh=0.5) -> pd.DataFrame:
    df = df.loc[:, df.isnull().mean() < col_thresh]
    mask = df.isnull().mean(axis=1) < row_thresh
    df = df.loc[mask]
    return df


# -----------------------------
# 4. Group-wise Imputation
# -----------------------------
def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    df_imputed = df.copy()

    # --- Group 1: Linear Interpolation ---
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

    df_imputed[numeric_cols] = (
        df_imputed.groupby("Country Name")[numeric_cols]
        .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )

    # --- Group 2: Median + Iterative ---
    group2 = [
        'GDP growth (annual %)',
        'Inflation, consumer prices (annual %)',
        'Real interest rate (%)'
    ]

    for country in df_imputed['Country Name'].unique():
        mask = df_imputed['Country Name'] == country
        sub = df_imputed.loc[mask, group2]

        valid_cols = sub.columns[sub.notnull().any()].tolist()
        if len(valid_cols) == 0:
            continue

        imp = SimpleImputer(strategy='median')
        df_imputed.loc[mask, valid_cols] = imp.fit_transform(sub[valid_cols])

    iter_imp = IterativeImputer(
        estimator=RandomForestRegressor(random_state=42),
        max_iter=10,
        random_state=42
    )
    df_imputed[group2] = iter_imp.fit_transform(df_imputed[group2])

    # --- Group 3: KNN ---
    group3 = [
        'Exports of goods and services (current US$)',
        'Imports of goods and services (current US$)'
    ]

    knn = KNNImputer(n_neighbors=5)
    df_imputed[group3] = knn.fit_transform(df_imputed[group3])

    # --- Final fallback ---
    df_imputed.fillna(df_imputed.median(numeric_only=True), inplace=True)

    return df_imputed


# -----------------------------
# 5. Full Pipeline
# -----------------------------
def preprocess_pipeline(filepath: str) -> pd.DataFrame:
    df = load_data(filepath)
    df = reshape_data(df)
    df = prune_missing(df)
    df = impute_data(df)

    # Drop invalid rows
    df = df.dropna(subset=["Time"])

    return df
