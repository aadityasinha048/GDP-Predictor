import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple,cast
from src.clustering import apply_cluster


# -----------------------------
# 1. Target Transformation
# -----------------------------
def transform_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log transform GDP
    df["GDP_log"] = np.log1p(df["GDP (current US$)"])

    return df


# -----------------------------
# 2. Encode Country
# -----------------------------

def encode_country(df: pd.DataFrame) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    df = df.copy()
    encoder = OrdinalEncoder()
    encoded = encoder.fit_transform(df[["Country Name"]])
    encoded = encoded.astype(float)  # ensure numeric

    df["Country_Code"] = pd.Series(encoded[:, 0], index=df.index)
    return df, encoder


# -----------------------------
# 3. Feature Groups
# -----------------------------
def get_feature_groups():
    group_economic = [
        'Exports of goods and services (current US$)',
        'Imports of goods and services (current US$)',
        'External debt stocks, total (DOD, current US$)',
        'Gross capital formation (current US$)',
        'General government final consumption expenditure (current US$)',
        'Services, value added (current US$)',
        'Industry (including construction), value added (current US$)',
        'Agriculture, forestry, and fishing, value added (current US$)',
        'Total reserves (includes gold, current US$)'
    ]

    group_percentage = [
        'GDP growth (annual %)',
        'Inflation, consumer prices (annual %)',
        'Real interest rate (%)',
        'Unemployment, total (% of total labor force)',
        'Gross savings (% of GDP)',
        'Foreign direct investment, net inflows (% of GDP)'
    ]

    group_social = [
        'Population, total',
        'Urban population',
        'Life expectancy at birth, total (years)',
        'Individuals using the Internet (% of population)'
    ]

    group_financial = [
        'Foreign direct investment, net inflows (BoP, current US$)',
        'Current account balance (BoP, current US$)',
        'Official exchange rate (LCU per US$, period average)'
    ]

    return group_economic, group_percentage, group_social, group_financial


# -----------------------------
# 4. Scaling Pipeline
# -----------------------------
def build_scaling_pipeline(df: pd.DataFrame):
    group_economic, group_percentage, group_social, group_financial = get_feature_groups()

    def filter_existing(columns, df):
        return [col for col in columns if col in df.columns]


    group_economic = filter_existing(group_economic, df)
    group_percentage = filter_existing(group_percentage, df)
    group_social = filter_existing(group_social, df)
    group_financial = filter_existing(group_financial, df)


    column_transformer = ColumnTransformer(transformers=[
        ('economic', PowerTransformer(), group_economic),
        ('percentage', StandardScaler(), group_percentage),
        ('social', MinMaxScaler(), group_social),
        ('financial', RobustScaler(), group_financial)
    ], remainder='drop')

    pipeline = Pipeline([
        ('scaler', column_transformer)
    ])

    return pipeline


# -----------------------------
# 5. Prepare Final Features
# -----------------------------
def prepare_features(df: pd.DataFrame, scaler=None, fit=False):
    df = transform_target(df)

    df, encoder = encode_country(df)
    df = cast(pd.DataFrame, df)

    # Separate target
    y = df["GDP_log"]

    # Drop unused columns
    X = df.drop(columns=[
        "GDP (current US$)",
        "GDP_log",
        "Country Name"
    ]).copy()
    X=cast(pd.DataFrame, X)

    # Build scaling pipeline
    if scaler is None:
        scaler = build_scaling_pipeline(df)
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    feature_names = scaler.get_feature_names_out()
    # Create DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # 🔥 CLEAN FEATURE NAMES (IMPORTANT)
    X_scaled_df.columns = [
        col.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
        .replace("$", "usd")
        .replace(",", "")
        for col in X_scaled_df.columns
    ]

    # Add country code back
    X_scaled_df["Country_Code"] = df["Country_Code"].values
    # =============================
    # 🔥 PHASE 6: ECONOMIC CONSISTENCY FEATURES
    # =============================

    # Trade contribution (Exports - Imports)
    if (
        "economic__Exports_of_goods_and_services_current_USusd" in X_scaled_df.columns
        and "economic__Imports_of_goods_and_services_current_USusd" in X_scaled_df.columns
    ):
        X_scaled_df["trade_contribution"] = (
            X_scaled_df["economic__Exports_of_goods_and_services_current_USusd"]
            - X_scaled_df["economic__Imports_of_goods_and_services_current_USusd"]
        )

    # Economic activity (Production side)
    if (
        "economic__Agriculture_forestry_and_fishing_value_added_current_USusd" in X_scaled_df.columns
        and "economic__Gross_capital_formation_current_USusd" in X_scaled_df.columns
    ):
        X_scaled_df["economic_activity"] = (
            X_scaled_df["economic__Agriculture_forestry_and_fishing_value_added_current_USusd"]
            + X_scaled_df["economic__Gross_capital_formation_current_USusd"]
        )

    # GDP proxy signal (structure-aware feature)
    if "economic_activity" in X_scaled_df.columns and "trade_contribution" in X_scaled_df.columns:
        X_scaled_df["gdp_proxy_signal"] = (
            X_scaled_df["economic_activity"]
            + X_scaled_df["trade_contribution"]
    )
    # 🔥 Interaction Features
    X_scaled_df["trade_balance"] = (
        X_scaled_df["economic__Exports_of_goods_and_services_current_USusd"]
        - X_scaled_df["economic__Imports_of_goods_and_services_current_USusd"]
    )

    # 🔥 NORMALIZED PER CAPITA OUTPUT
    X_scaled_df["per_capita_output"] = (
        X_scaled_df["economic__Gross_capital_formation_current_USusd"]
        / (X_scaled_df["social__Population_total"] + 1e-6)
    )

    # Clip extreme values (VERY IMPORTANT)
    X_scaled_df["per_capita_output"] = np.clip(
        X_scaled_df["per_capita_output"], -10, 10
    )
    # Normalize economic consistency features
    for col in ["trade_contribution", "economic_activity"]:
        if col in X_scaled_df.columns:
            X_scaled_df[col] = np.clip(X_scaled_df[col], -10, 10)
    
    # =============================
    # 🔥 PHASE 10: COUNTRY CLUSTERING
    # =============================

    cluster_features = [
        "economic__Exports_of_goods_and_services_current_USusd",
        "economic__Imports_of_goods_and_services_current_USusd",
        "social__Urban_population"
    ]

    available_features = [f for f in cluster_features if f in X_scaled_df.columns]

    if len(available_features) >= 2:
        cluster_data = X_scaled_df[available_features].copy()
        cluster_data = cluster_data.fillna(cluster_data.mean()).fillna(0)

        try:
            X_scaled_df["country_cluster"] = apply_cluster(cluster_data)
        except:
            X_scaled_df["country_cluster"] = 0
        
    return X_scaled_df, y, scaler, encoder