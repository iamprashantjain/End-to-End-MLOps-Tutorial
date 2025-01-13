import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Set the MLflow tracking URI to the 'mlruns' folder
mlflow.set_tracking_uri('file:///D:/End-to-End-MLOps-Tutorial/5_end_to_end_mlflow/mlruns')
mlflow.set_experiment('breast-cancer-rf-hp')

# Set the artifact location to the 'mlartifacts' folder
artifact_uri = 'file:///D:/End-to-End-MLOps-Tutorial/5_end_to_end_mlflow/mlartifacts'
os.environ['MLFLOW_ARTIFACT_URI'] = artifact_uri  # Setting environment variable

# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Start the MLflow run
with mlflow.start_run() as parent_run:
    grid_search.fit(X_train, y_train)

    # Log the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    # Log the grid search results
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child_run:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Log training and test data to 'mlartifacts' folder
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_file_path = "train_data.csv"
    train_df.to_csv(train_file_path, index=False)

    # Here, explicitly set the artifact path to save it under 'mlartifacts'
    mlflow.log_artifact(train_file_path, artifact_path=os.path.join('mlartifacts', 'training'))

    test_df = X_test.copy()
    test_df['target'] = y_test
    test_file_path = "test_data.csv"
    test_df.to_csv(test_file_path, index=False)

    # Save test data to 'mlartifacts/testing'
    mlflow.log_artifact(test_file_path, artifact_path=os.path.join('mlartifacts', 'testing'))

    # Log the best model to 'mlartifacts' folder
    input_example = X_train.iloc[0].to_numpy().reshape(1, -1)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest", input_example=input_example)

    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")
