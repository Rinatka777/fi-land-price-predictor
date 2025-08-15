import numpy as np
import json
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import os

df = pd.read_csv("../data/processed/finnish_land_clean.csv")

features_with_index = ["reaalihintaindeksi", "keskipinta_ala", "kauppojen_lkm", "year", "q_num"]
features_without_index = ["keskipinta_ala", "kauppojen_lkm", "year", "q_num"]
target = "neliöhinta"

df = df.sort_values(by=["year", "q_num"])
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

def min_max_scaler(train_df, test_df, feature_names):
    train_copy = train_df.copy(deep=True)
    test_copy = test_df.copy(deep=True)
    min_max_params = {}
    for col in feature_names:
        min_val = train_copy[col].min()
        max_val = train_copy[col].max()
        if max_val - min_val == 0:
            train_copy[col] = 0
            test_copy[col] = 0
            min_max_params[col] = (float(min_val), float(max_val))
        else:
            train_copy[col] = (train_copy[col] - min_val) / (max_val - min_val)
            test_copy[col] = (test_copy[col] - min_val) / (max_val - min_val)
            min_max_params[col] = (float(min_val), float(max_val))
    return min_max_params, train_copy, test_copy

def train_and_evaluate(x_train, y_train, x_test, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return model, mae, rmse, r2

def run_experiment(feature_list, tag):
    min_max_params, train_scaled, test_scaled = min_max_scaler(train_df, test_df, feature_list)
    x_train = train_scaled[feature_list]
    y_train = train_scaled[target]
    x_test = test_scaled[feature_list]
    y_test = test_scaled[target]
    model, mae, rmse, r2 = train_and_evaluate(x_train, y_train, x_test, y_test)
    result = {"name": tag, "mae": mae, "rmse": rmse, "r2": r2}
    return model, min_max_params, result

model_a, params_a, res_a = run_experiment(features_with_index, "with_index")
model_b, params_b, res_b = run_experiment(features_without_index, "without_index")

print("A:", res_a)
print("B:", res_b)

best = res_a if res_a["rmse"] <= res_b["rmse"] else res_b
selected_features = features_with_index if best["name"] == "with_index" else features_without_index
selected_model = model_a if best["name"] == "with_index" else model_b
selected_params = params_a if best["name"] == "with_index" else params_b

print("Selected:", {"name": best["name"], "rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"]})

os.makedirs("../models", exist_ok=True)

joblib.dump(selected_model, "../models/model.joblib")

with open("../models/scaler_params.json", "w") as f:
    json.dump(selected_params, f)

with open("../models/features.json", "w") as f:
    json.dump(selected_features, f)

metadata = {
    "selected_model": best["name"],
    "metrics": best,
    "created_at": datetime.now().isoformat(),
    "data_rows": len(df)
}
with open("../models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
