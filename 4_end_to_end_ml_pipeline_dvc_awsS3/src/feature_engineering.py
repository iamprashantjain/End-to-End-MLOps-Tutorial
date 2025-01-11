import sys
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

from logger_config import logger
from custom_exception import CustomException


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.info('Unexpected error occurred while loading the data: %s', e)
        raise CustomException(e,sys)


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TfIdf to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.info('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.info('Error during Bag of Words transformation: %s', e)
        raise CustomException(e,sys)

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info('Data saved to %s', file_path)
    except Exception as e:
        logger.info('Unexpected error occurred while saving the data: %s', e)
        raise CustomException(e,sys)

def main():
    try:
        max_features = 50
        
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.info('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()