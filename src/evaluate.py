import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Performance:")
    print(f"MAE  : {mae}")
    print(f"RMSE : {rmse}")
    print(f"R²   : {r2}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }