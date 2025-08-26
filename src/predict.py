# src/predict.py

import argparse
import pandas as pd
import joblib
from features import add_features, build_preprocessor

def predict(model, input_path, output_path):
    # 1. Load and preprocess data
    data = pd.read_csv(input_path)
    data = add_features(data)

    # Drop unnecessary columns for prediction
    X = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # 2. Predict
    preds = model.predict(X)

    # 3. Save predictions
    output = pd.DataFrame({"PassengerId": data["PassengerId"], "Survived": preds})
    output.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to save predictions")
    args = parser.parse_args()

    # Load trained model (pipeline includes preprocessing)
    model = joblib.load("models/titanic_best_model.pkl")
    predict(model, args.input, args.output)
