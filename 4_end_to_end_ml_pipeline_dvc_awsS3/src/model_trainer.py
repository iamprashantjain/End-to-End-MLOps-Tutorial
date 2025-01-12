import yaml
import sys
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

from logger_config import logger
from custom_exception import CustomException



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
    
    
    
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.info('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except Exception as e:
        logger.info('Unexpected error occurred while loading the data: %s', e)
        raise CustomException(e,sys)

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.info('Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.info('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.info('Model training completed')
        
        return clf
    except Exception as e:
        logger.info('Error during model training: %s', e)
        raise CustomException(e,sys)


def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info('Model saved to %s', file_path)
    except Exception as e:
        logger.info('Error occurred while saving the model: %s', e)
        raise CustomException(e,sys)

def main():
    try:
        params = load_params('params.yaml')['model_trainer']
        # params = {'n_estimators':25, 'random_state':2}
        
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.info('Failed to complete the model building process: %s', e)
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()