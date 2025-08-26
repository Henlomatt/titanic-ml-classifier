# src/train.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from features import add_features, build_preprocessor

# 1. Load data
data = pd.read_csv("data/titanic.csv")

# 2. Feature engineering
data = add_features(data)

# 3. Group rare titles
rare_titles = data["Title"].value_counts()[data["Title"].value_counts() < 10].index
data["Title"] = data["Title"].replace(rare_titles, "Rare")

# 4. Define features and target
X = data.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
y = data["Survived"]

# 5. Preprocessing pipeline from features.py
preprocessor = build_preprocessor(X)

# 6. Model pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Hyperparameter tuning
param_grid = {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
grid_search.fit(X_train, y_train)

# 9. Best model
best_model = grid_search.best_estimator_

# 10. Evaluate
test_accuracy = best_model.score(X_test, y_test)
cv_scores = cross_val_score(best_model, X, y, cv=5)

print("Best parameters:", grid_search.best_params_)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 11. Save model
os.makedirs("models", exist_ok=True)
model_path = "models/titanic_best_model.pkl"
joblib.dump(best_model, model_path)
print(f"✅ Model saved to {model_path}")
