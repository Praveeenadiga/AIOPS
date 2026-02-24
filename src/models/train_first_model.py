"""
First MLflow Experiment - AIOps & MLOps Training
Week 1, Session 1

This script demonstrates:
1. Loading data
2. Training a simple model
3. Logging everything to MLflow
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

def main():
    # Set experiment name
    mlflow.set_experiment("week1-first-experiment")

    # Load data
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
    }

    # Start MLflow run
    with mlflow.start_run(run_name="first-rf-experiment"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("test_size", 0.2)

        # Train model
        print("Training Random Forest...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "random-forest-model")

        # Log dataset info as artifact
        dataset_info = f"""Dataset: Iris
Samples: {len(X)}
Features: {len(X.columns)}
Classes: {len(np.unique(y))}
Train size: {len(X_train)}
Test size: {len(X_test)}
"""
        with open("dataset_info.txt", "w") as f:
            f.write(dataset_info)
        mlflow.log_artifact("dataset_info.txt")

        # Print results
        print("\n===== EXPERIMENT RESULTS =====")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print("==============================")
        print("\nExperiment logged to MLflow!")
        print("Run 'mlflow ui' to view results in the browser.")

if __name__ == "__main__":
    main()
