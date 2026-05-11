import pandas as pd
import numpy as np
import joblib

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import prepare_features


# -----------------------------
# Load Saved Artifacts
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")
selected_features = joblib.load("models/features.pkl")

def predict_gdp_from_features(X):
    y_pred_log = model.predict(X)
    return np.expm1(y_pred_log)

def explain_scenario(X, X_new):
    diff = X_new - X

    print("\n📊 Feature Impact (Change):")
    print(diff)

    print("\n📈 Absolute Impact Ranking:")
    impact = diff.abs().mean().sort_values(ascending=False)
    print(impact)
# -----------------------------
# Predict Function
# -----------------------------
def predict_gdp(input_df: pd.DataFrame):
    print("\n📥 Input to predict_gdp():")
    print(input_df.head())
    # Apply feature engineering
    X, _, _, _ = prepare_features(input_df)
    print("\n🧠 Features after engineering:")
    print(X.head())
    # Select same features
    X = X[selected_features]
    print("\n🎯 Features used for prediction:")
    print(X.head())
    # Predict log GDP
    y_pred_log = model.predict(X)

    # Convert back to original GDP scale
    y_pred = np.expm1(y_pred_log)

    return y_pred

def simulate_scenario(df: pd.DataFrame, changes: dict):
    df_copy = df.copy()
    print("\n⚙️ Applying Scenario Changes:")
    print("Changes:", changes)
    for feature, change in changes.items():
        # Apply only if feature exists in original data
        if feature in df_copy.columns:
            df_copy[feature] = df_copy[feature] * (1 + change)
    print("\n📊 Modified Data (after scenario):")
    print(df_copy.head())
    return df_copy

def predict_scenario(df: pd.DataFrame, changes: dict):

    # Step 1: Convert raw data → features
    X, _, _, _ = prepare_features(df, scaler=scaler, fit=False)

    print("\n🧠 Original Features:")
    print(X.head())

    # Step 2: Select trained features
    X = X[selected_features]

    # Step 3: Baseline prediction
    base_pred = predict_gdp_from_features(X)

    # Step 4: Apply scenario DIRECTLY on features

    print("\n⚙️ Applying Scenario on Features:")
    print("Changes:", changes)

    # Step 4: Apply scenario on RAW data
    df_new = df.copy()

    print("\n⚙️ Applying Scenario on RAW data:")

    # 🔥 Dynamic mapping from engineered → raw
    feature_map = {
        "economic__Exports_of_goods_and_services_current_USusd":
            "Exports of goods and services (current US$)",

        "economic__Imports_of_goods_and_services_current_USusd":
            "Imports of goods and services (current US$)",

        "economic__Gross_capital_formation_current_USusd":
            "Gross capital formation (current US$)",

        "social__Individuals_using_the_Internet_percent_of_population":
            "Individuals using the Internet (% of population)",

        "social__Urban_population":
            "Urban population",

        "economic__Total_reserves_includes_gold_current_USusd":
            "Total reserves (includes gold, current US$)"
    }


    for eng_feature, change in changes.items():

        if eng_feature in feature_map:
            raw_feature = feature_map[eng_feature]

            if raw_feature in df_new.columns:
                sensitivity = 1.5  # you can tune this

                df_new[raw_feature] = df_new[raw_feature] * (1 + change * sensitivity)
    
    print("\n🔍 RAW DATA CHANGE CHECK:")
    print(df_new.head())

    # Recompute features AFTER change
    X_new, _, _, _ = prepare_features(df_new, scaler=scaler, fit=False)
    X_new = X_new[selected_features]
    
    # 🔍 Explain impact
    explain_scenario(X, X_new)

    # =============================
    # 🔍 ECONOMIC CONSISTENCY CHECK
    # =============================
    print("\n🧠 Economic Consistency Check:")

    for col in ["trade_contribution", "economic_activity", "gdp_proxy_signal"]:
        if col in X_new.columns:
            print(f"{col}:")
            print(X_new[col].values)

    print("\n📊 Modified Features:")
    print(X_new.head())

    if "country_cluster" in X.columns:
        print("\n🌍 Country Clusters:")
        print(X["country_cluster"].values)

    # Step 5: New prediction
    new_pred = predict_gdp_from_features(X_new)

    return base_pred, new_pred

def run_sensitivity(df, feature):

    values = [-0.2, -0.1, 0, 0.1, 0.2]

    print(f"\n📊 Sensitivity Analysis for {feature}")

    for v in values:
        scenario = {feature: v}
        base, new = predict_scenario(df, scenario)

        print(f"Change {v*100:+.0f}% → ΔGDP = {(new - base).mean():.2e}")

def get_top_drivers(X, model, top_n=5):

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Model does not support feature importance")
        return

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    })

    feature_importance = feature_importance.sort_values(
        by="importance", ascending=False
    )

    print("\n🔥 Top GDP Drivers:")
    print(feature_importance.head(top_n))

def find_best_policy(df, features_to_test):

    # 🔥 Map engineered → raw features
    feature_map = {
        "economic__Exports_of_goods_and_services_current_USusd":
            "Exports of goods and services (current US$)",

        "economic__Imports_of_goods_and_services_current_USusd":
            "Imports of goods and services (current US$)",

        "economic__Gross_capital_formation_current_USusd":
            "Gross capital formation (current US$)",

        "social__Individuals_using_the_Internet_percent_of_population":
            "Individuals using the Internet (% of population)",

        "social__Urban_population":
            "Urban population",

        "economic__Total_reserves_includes_gold_current_USusd":
            "Total reserves (includes gold, current US$)"
    }

    results = []

    for feature in features_to_test:

        if feature in feature_map:
            scenario = {feature: 0.1}
        else:
            continue

        base, new = predict_scenario(df, scenario)

        impact = (new - base).mean()

        results.append({
            "feature": feature,
            "impact": impact
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="impact", ascending=False)

    print("\n🏆 Best Policy Options:")
    print(results_df)

    return results_df

def run_multi_scenario(df, scenario):

    print("\n🌍 Running Multi-variable Scenario...")

    base, new = predict_scenario(df, scenario)

    print("\n📊 Multi Scenario Result:")
    print("Change:", scenario)
    print("ΔGDP:", (new - base))

    return base, new
# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    df = preprocess_pipeline("data/raw/World_data_GDP.csv")
    sample = df.sample(3, random_state=42)

    scenario = {
    "economic__Exports_of_goods_and_services_current_USusd": 0.05,
    "economic__Imports_of_goods_and_services_current_USusd": -0.02
}
    
    base, new = predict_scenario(sample, scenario)

    print("\nBaseline GDP:", base)
    print("Scenario GDP:", new)
    print("\n🔍 Difference:")
    print(new - base)
    run_sensitivity(
    sample,
    "economic__Exports_of_goods_and_services_current_USusd"
)
    # =============================
    # 🔥 PHASE 11: DECISION INTELLIGENCE
    # =============================

    # 1. Top Drivers
    X, _, _, _ = prepare_features(sample)
    X = X[selected_features]
    get_top_drivers(X, model)


    # 2. Best Policy Finder
    features_to_test = [
        "economic__Exports_of_goods_and_services_current_USusd",
        "economic__Imports_of_goods_and_services_current_USusd",
        "economic__Gross_capital_formation_current_USusd",
        "social__Individuals_using_the_Internet_percent_of_population"
    ]

    find_best_policy(sample, features_to_test)


    # 3. Multi-variable Scenario
    multi_scenario = {
        "economic__Exports_of_goods_and_services_current_USusd": 0.1,
        "economic__Gross_capital_formation_current_USusd": 0.05
    }

    run_multi_scenario(sample, multi_scenario)