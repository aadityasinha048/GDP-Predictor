from src.train import run_training_pipeline
from src.predict import predict_gdp
from src.data_preprocessing import preprocess_pipeline

def main():
    print("🚀 Running Full GDP Pipeline")

    # Train model
    run_training_pipeline()

    # Test prediction
    df = preprocess_pipeline("data/raw/World_data_GDP.csv")
    sample = df.sample(3, random_state=42)

    preds = predict_gdp(sample)

    print("\nSample Predictions:")
    print(preds)

if __name__ == "__main__":
    main()