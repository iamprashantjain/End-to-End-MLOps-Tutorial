import yaml
import sys
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

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

      
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.info('Data loaded from %s', data_url)
        return df
    except Exception as e:
        logger.info('Error loading data from %s: %s', data_url, e)
        raise CustomException(e,sys)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # Handle missing or unnecessary columns gracefully
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.info('Data preprocessing completed')
        return df
    except Exception as e:
        logger.info('Error during preprocessing: %s', e)
        raise CustomException(e,sys)

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.info('Error saving data to %s: %s', data_path, e)
        raise CustomException(e,sys)

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(data_url=data_path)
        
        # Preprocess data
        final_df = preprocess_data(df)
        
        # Split into train and test datasets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # Save train and test data
        save_data(train_data, test_data, data_path='./data')
        
    except Exception as e:
        logger.info(f"Unexpected error: {e}")
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()