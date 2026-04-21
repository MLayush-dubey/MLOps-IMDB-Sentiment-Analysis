import numpy as np 
import pandas as pd 
import pickle 
from sklearn.linear_model import LogisticRegression
import yaml 
from src.logger import logging 


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info("Data Loaded successfully")
        return df 
    except pd.errors.ParserError as e:
        logging.error("Error parsing CSV file: %s", e)
        raise 
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise 


def train_model(X_train:np.ndarray, y_train:np.ndarray) -> LogisticRegression:
    """Train the logistic regression model"""
    try:
        clf = LogisticRegression() 
        clf.fit(X_train, y_train)
        logging.info("Model trained successfully")
        return clf 
    except Exception as e:
        logging.error("Error training model: %s", e)
        raise 


def save_model(model: LogisticRegression, model_path: str) -> None:
    """Save the trained model"""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info("Model saved to: %s", model_path)
    except Exception as e:
        logging.error("Error saving model: %s", e)
        raise 


def main():
    try:

        train_data = load_data("./data/processed/train_bow.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values 

        clf = train_model(X_train, y_train)

        save_model(clf, "models/model.pkl")
    except Exception as e:
        logging.error("Failed to complete the model building process: %s", e)
        raise 


if __name__ == "__main__":
    main()