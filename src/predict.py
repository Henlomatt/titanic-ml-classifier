import argparse
import pandas as pd
import joblib
import os
from features import add_features

def load_model(model_path: str):
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def predict(model, input_path, output_path):
    data = pd.read_csv(input_path)
    data = add_features(data)   # reuse function
    preds = model.predict(data)
    data['Prediction'] = preds
    data[['PassengerId', 'Prediction']].to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    
    # Ensure no PassengerId/Survived column issues
    if "Survived" in data.columns:
        data = data.drop(columns=["Survived"])

    if output_csv:
        data.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print(data[["Prediction"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with trained Titanic model")
    parser.add_argument("--model", type=str, default="models/titanic_best_model.pkl", help="Path to trained model")
    parser.add_argument("--input", type=str, required=True, help="CSV file with passenger data")
    parser.add_argument("--output", type=str, help="Optional: file to save predictions")

    args = parser.parse_args()

    model = load_model(args.model)
    predict(model, args.input, args.output)
