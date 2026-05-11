import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from collections import Counter


# -----------------------------
# 1. Correlation Selection
# -----------------------------
def correlation_selection(X: pd.DataFrame, y: pd.Series, threshold=0.1):
    corr = X.corrwith(y).abs()
    selected = corr[corr > threshold].index.tolist()
    return selected


# -----------------------------
# 2. Mutual Information
# -----------------------------
def mutual_info_selection(X: pd.DataFrame, y: pd.Series):
    mi = mutual_info_regression(X, y)
    mi_series = pd.Series(mi, index=X.columns)

    threshold = mi_series.quantile(0.7)
    selected = mi_series[mi_series > threshold].index.tolist()

    return selected


# -----------------------------
# 3. Random Forest Importance
# -----------------------------
def rf_selection(X: pd.DataFrame, y: pd.Series):
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=X.columns)

    threshold = importance.quantile(0.4)
    selected = importance[importance > threshold].index.tolist()

    return selected, importance


# -----------------------------
# 4. RFE Selection
# -----------------------------
def rfe_selection(X: pd.DataFrame, y: pd.Series, n_features=12):
    model = LinearRegression()

    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)

    selected = X.columns[rfe.support_].tolist()

    return selected


# -----------------------------
# 5. Voting System
# -----------------------------
def voting_selection(corr_sel, mi_sel, rf_sel, rfe_sel, min_votes=1):
    all_features = corr_sel + mi_sel + rf_sel + rfe_sel

    counts = Counter(all_features)

    selected = [feat for feat, count in counts.items() if count >= min_votes]

    return selected


# -----------------------------
# 6. Remove Highly Correlated Features
# -----------------------------
def remove_multicollinearity(X: pd.DataFrame, threshold=0.9):
    corr_matrix = X.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    return [col for col in X.columns if col not in to_drop]


# -----------------------------
# 7. Final Feature Selection Pipeline
# -----------------------------
def select_features(X: pd.DataFrame, y: pd.Series, top_n=12):

    # Step 1: individual methods
    corr_sel = correlation_selection(X, y)
    mi_sel = mutual_info_selection(X, y)
    rf_sel, importance = rf_selection(X, y)
    rfe_sel = rfe_selection(X, y, n_features=top_n)

    # Step 2: voting
    voted_features = voting_selection(corr_sel, mi_sel, rf_sel, rfe_sel)
    if len(voted_features) < top_n:
        fallback = importance.sort_values(ascending=False).head(top_n).index.tolist()
        voted_features = list(set(voted_features + fallback))

    # Step 3: reduce X
    X_selected = X[voted_features]

    # Step 4: remove multicollinearity
    final_features = remove_multicollinearity(X_selected)

    # Step 5: keep top N based on RF importance
    final_features = (
        importance.loc[final_features]
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    # 🔥 FORCE INCLUDE COUNTRY CLUSTER
    if "country_cluster" in X.columns and "country_cluster" not in final_features:
        final_features.append("country_cluster")

    return final_features