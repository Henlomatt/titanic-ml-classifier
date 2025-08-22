# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
data = pd.read_csv("data/titanic.csv")

# 2. Basic feature engineering
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
data["Title"] = data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

# Reduce rare titles
rare_titles = data["Title"].value_counts()[data["Title"].value_counts() < 10].index
data["Title"] = data["Title"].replace(rare_titles, "Rare")

# 3. Define features and target
X = data.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
y = data["Survived"]

# 4. Preprocessing
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 5. Model pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Hyperparameter tuning
param_grid = {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
grid_search.fit(X_train, y_train)

# 8. Best model
best_model = grid_search.best_estimator_

# 9. Evaluate
test_accuracy = best_model.score(X_test, y_test)
cv_scores = cross_val_score(best_model, X, y, cv=5)

print("Best parameters:", grid_search.best_params_)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
