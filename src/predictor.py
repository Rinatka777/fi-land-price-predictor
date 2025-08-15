import json
import joblib
import pandas as pd
from datetime import datetime
import os

class LandPricePredictor:
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model_path = os.path.join(root_dir, "models", "model.joblib")
        params_path = os.path.join(root_dir, "models", "scaler_params.json")
        features_path = os.path.join(root_dir, "models", "features.json")
        log_path = os.path.join(root_dir, "logs", "predictions_land.csv")

        self.model = joblib.load(model_path)

        with open(params_path, "r") as f:
            self.scaler_params = json.load(f)

        with open(features_path, "r") as f:
            self.features = json.load(f)

        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def scale_features(self, row):
        scaled = {}
        for feature in self.features:
            min_val, max_val = self.scaler_params[feature]
            if max_val - min_val == 0:
                scaled[feature] = 0
            else:
                scaled[feature] = (row[feature] - min_val) / (max_val - min_val)
        return scaled

    def predict(self, input_dict):
        scaled_input = self.scale_features(input_dict)
        X = pd.DataFrame([scaled_input])[self.features]
        prediction = self.model.predict(X)[0]
        return prediction

    def run_cli(self):
        print("\n=== Land Price Predictor ===\n")
        input_dict = {}
        for feature in self.features:
            val = float(input(f"Enter value for {feature}: "))
            input_dict[feature] = val

        pred = self.predict(input_dict)
        print(f"\nPredicted price (EUR/m²): {pred:.2f}")

        self.log_prediction(input_dict, pred)
        print(f"Prediction logged to {self.log_path}")

    def log_prediction(self, input_dict, prediction):
        log_row = {
            "timestamp": datetime.now().isoformat(),
            **input_dict,
            "prediction": prediction
        }
        if os.path.exists(self.log_path):
            df = pd.read_csv(self.log_path)
            df = pd.concat([df, pd.DataFrame([log_row])], ignore_index=True)
        else:
            df = pd.DataFrame([log_row])
        df.to_csv(self.log_path, index=False)


if __name__ == "__main__":
    predictor = LandPricePredictor()
    predictor.run_cli()
