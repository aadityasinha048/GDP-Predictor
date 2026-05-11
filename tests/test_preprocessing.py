import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_pipeline

# Path to dataset
file_path = "data/raw/World_data_GDP.csv"

df = preprocess_pipeline(file_path)

print("✅ Shape:", df.shape)
print("\n✅ Columns:", df.columns.tolist()[:10])
print("\n✅ Sample Data:")
print(df.head())

print("\n✅ Missing Values:", df.isna().sum().sum())