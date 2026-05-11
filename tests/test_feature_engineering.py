import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import prepare_features

df = preprocess_pipeline("data/raw/World_data_GDP.csv")

X, y, scaler, encoder = prepare_features(df)

print("✅ X shape:", X.shape)
print("✅ y shape:", y.shape)
print("✅ Sample features:")
print(X.head())