import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import prepare_features
from src.feature_selection import select_features

df = preprocess_pipeline("data/raw/World_data_GDP.csv")

X, y, _, _ = prepare_features(df)

selected_features = select_features(X, y)

print("✅ Selected Features:", selected_features)
print("✅ Feature Count:", len(selected_features))