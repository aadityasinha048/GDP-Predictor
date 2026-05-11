import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
import lightgbm as lgb

import joblib

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import prepare_features
from src.feature_selection import select_features
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from src.clustering import train_cluster

# -----------------------------
# 1. Train Models
# -----------------------------
def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    }

    results = []

    best_model = None
    best_score = -np.inf
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        print(f"{name} → CV R2: {cv_scores.mean():.4f}")

        y_pred = model.predict(X_test)
        print(f"\n{name} Sample Predictions:")
        print("Pred:", y_pred[:5])
        print("Actual:", y_test[:5].values)
        residuals = y_test - y_pred

        print(f"{name} Residual Mean: {residuals.mean():.4f}")
        print(f"{name} Residual Std: {residuals.std():.4f}")

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(
        np.expm1(y_test), np.expm1(y_pred)
    )

        results.append({
            "Model": name,
            "MAE": mae,
            "R2 Score": r2,
            "RMSE": rmse,
            "MAPE": mape
        })

        print(f"{name} → MAE: {mae:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)

    return best_model, best_model_name, results_df


# -----------------------------
# 2. Full Training Pipeline
# -----------------------------
def run_training_pipeline(data_path="data/raw/World_data_GDP.csv"):

    print("🔄 Loading and preprocessing data...")
    df = preprocess_pipeline(data_path)

    print("⚙️ Feature engineering...")
    X, y, scaler, encoder = prepare_features(df, fit=True)

    print("🎯 Feature selection...")
    selected_features = select_features(X, y)

    # 🔥 FORCE INCLUDE ECONOMIC DRIVERS
    force_features = [
        "economic__Exports_of_goods_and_services_current_USusd",
        "economic__Imports_of_goods_and_services_current_USusd"
    ]

    for f in force_features:
        if f in X.columns and f not in selected_features:
            selected_features.append(f)
    if "Country_Code" in selected_features:
        selected_features.remove("Country_Code")
    
    X = X[selected_features]
    print("\nFinal Features Used:")
    print(selected_features)
    print(f"✅ Using {len(selected_features)} features")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training models...")
    best_model, best_model_name, results_df = train_models(
        X_train, X_test, y_train, y_test
    )

    print(f"\n🏆 Best Model: {best_model_name}")

    cluster_features = [
        "economic__Exports_of_goods_and_services_current_USusd",
        "economic__Imports_of_goods_and_services_current_USusd",
        "social__Urban_population"
    ]

    available_cluster_features = [f for f in cluster_features if f in X.columns]

    if len(available_cluster_features) >= 2:
        train_cluster(X[available_cluster_features])
    # 🔍 Feature Importance
    if hasattr(best_model, "feature_importances_"):
        import matplotlib.pyplot as plt

        importances = best_model.feature_importances_
        features = selected_features

        sorted_idx = np.argsort(importances)

        plt.figure()
        plt.barh(np.array(features)[sorted_idx], importances[sorted_idx])
        plt.title("Feature Importance")
        plt.show()
    # -----------------------------
    # 3. Save Artifacts
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(selected_features, "models/features.pkl")

    results_df.to_csv("models/model_metrics.csv", index=False)

    print("💾 Models and metrics saved in /models")

    return best_model, results_df


# -----------------------------
# 3. Run Script
# -----------------------------
if __name__ == "__main__":
    run_training_pipeline()