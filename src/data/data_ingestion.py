import numpy as np 
import pandas as pd 
pd.set_option('future.no_silent_downcasting', True)  #basically warning dega before changing data(null values ya kuch bhi)

import os 
from sklearn.model_selection import train_test_split 
import logging 
import yaml 
from src.logger import logging 
from src.connections import s3_connection


def load_params(params_path: str) -> dict:
    """Load parameters from YAML file into a dictionary"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", params_path)
        return params 
    except FileNotFoundError:
        logging.error("File not found: %s", params_path)
        raise 
    except yaml.YAMLError as e:
        logging.error("YAML error: %s", e)
        raise 
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise 


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        df = pd.read_csv(data_url)
        logging.info("Data loaded successfully from: %s", data_url)
        return df 
    except pd.errors.ParserError as e:   #this occurrs when csv file is corrupt
        logging.error("Parser Error: %s", e)
        raise 
    except Exception as e:
        logging.error("Unexpected error occurred while loading the data: %s", e)
        raise 


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the loaded data"""
    try:
        logging.info("Pre-processing the data")
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].map({'positive': 1, 'negative': 0})
        return final_df
    except KeyError as e:
        logging.error("Missing column in the dataframe: %s", e)
        raise 
    except Exception as e:
        logging.error("Unexpected error occurred while preprocessing the data: %s", e)
        raise 


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str ) -> None:
    """Save the train and test data"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')   #jo data path denge usme raw banake usme save krenge 
        os.makedirs(raw_data_path, exist_ok = True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index = False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index = False)
        logging.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logging.error("Unexpected error occurred while saving the data: %s", e)
        raise 


def main():
    try:
        # params = load_params(params_path="params.yaml")
        # test_size = params['data_ingestion']['test_size']
        test_size = 0.2 

        df = load_data(data_url = "notebooks/data.csv")
        s3 = s3_connection.s3_operations(
            bucket_name = os.getenv("S3_BUCKET_NAME"),
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY"))
        #df = s3.fetch_file_from_s3("data.csv")

        final_df = preprocess_data(df) 
        train_data, test_data = train_test_split(final_df, test_size = test_size, random_state = 42)
        save_data(train_data, test_data, data_path = './data')
    except Exception as e:
        logging.error("Failed to complete the data ingestion process: %s", e)
        raise 


if __name__ == "__main__":
    main()
        