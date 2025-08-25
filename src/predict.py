import argparse
import pandas as pd
import joblib
from features import add_features

def predict(model, input_path, output_path):
    data = pd.read_csv(input_path)
    data = add_features(data)
    preds = model.predict(data)
    data['Prediction'] = preds
    data[['PassengerId', 'Prediction']].to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to save predictions")
    args = parser.parse_args()

    model = joblib.load("models/titanic_best_model.pkl")
    predict(model, args.input, args.output)
