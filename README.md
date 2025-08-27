# Titanic ML Classifier 🚢🧠

This repository contains a complete **Titanic Machine Learning project** including exploratory data analysis (EDA), feature engineering, model training, evaluation, and prediction. Perfect for showcasing ML skills on GitHub.

## Project Structure

titanic-ml-classifier/
├── data/ # Raw input data
├── notebooks/ # EDA and analysis
│ └── eda.ipynb
├── src/ # Source code
│ ├── features.py
│ ├── train.py
│ └── predict.py
├── models/ # Saved ML models
│ └── titanic_best_model.pkl
├── outputs/ # Generated predictions (ignored by Git)
│ └── .gitkeep
├── .gitignore
├── environment.yml # Conda environment
├── requirements.txt
├── README.md
└── setup_repo.sh

## Features

- **EDA:** Visualizations, missing value analysis, correlations, feature distributions.
- **Feature Engineering:** Family size, title extraction, IsAlone flag.
- **Models:** Logistic Regression, Random Forest, and more.
- **Pipeline:** Preprocessing, training, evaluation, and prediction in `src/`.
- **Predictions:** Generate CSV files using `predict.py` (stored in `outputs/`).

## Usage

### 1. Set up environment

```bash
conda env create -f environment.yml
conda activate titanic_env

2. Train model

python src/train.py

This will save the best trained model to models/titanic_best_model.pkl.

3. Make predictions

python src/predict.py --input data/test.csv --output outputs/predictions.csv

Predictions will be stored in the outputs/ folder (ignored by Git).

License

MIT License

## 📝 Notebooks

This repository includes interactive Jupyter notebooks demonstrating the full Titanic ML pipeline:

- **[EDA](notebooks/eda.ipynb)** – Exploratory Data Analysis: visualizations, missing values, feature distributions, and correlations.
- **[Model Evaluation](notebooks/model_evaluation.ipynb)** – Train and evaluate baseline and advanced models, including Logistic Regression and Random Forest, with performance metrics, confusion matrices, ROC curves, and feature importance analysis.
---



