import sys
import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from logger_config import logger
from custom_exception import CustomException

from dvclive import Live
import yaml


#read and return all params inside params.yaml file
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.info('Unexpected error: %s', e)
        raise CustomException(e,sys)


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.info('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logger.info('File not found: %s', file_path)
        raise CustomException(e,sys)

#load test data to evaluate
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.info('Unexpected error occurred while loading the data: %s', e)
        raise CustomException(e,sys)

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.info('Error during model evaluation: %s', e)
        raise CustomException(e,sys)

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', file_path)
    except Exception as e:
        logger.info('Error occurred while saving the metrics: %s', e)
        raise CustomException(e,sys)

def main():
    try:
        params = load_params(params_path='params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        
        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            #also logging all data in params.yaml file
            live.log_params(params)
              
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.info('Failed to complete the model evaluation process: %s', e)
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()