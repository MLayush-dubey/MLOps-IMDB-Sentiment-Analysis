import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pickle 
import json 
from src.logger import logging 
import mlflow 
import mlflow.sklearn
import os 
import dagshub


#Only for deployment
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token 
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token 

dagshub_url = "https://dagshub.com"
repo_owner = "MLayush-dubey"
repo_name = "MLOps-IMDB-Sentiment-Analysis"

#setup the mlflow tracking URI
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")



#------------------------------------------------------------------------
#Below code is for dev
mlflow.set_tracking_uri("https://dagshub.com/MLayush-dubey/MLOps-IMDB-Sentiment-Analysis.mlflow")
dagshub.init(repo_owner = "MLayush-dubey", repo_name = "MLOps-IMDB-Sentiment-Analysis", mlflow = True)


def load_model(file_path: str):
    try:    
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully from: %s", file_path)
        return model
    except FileNotFoundError:
        logging.error("Unexpected error occurred while loading the model: %s", file_path)
        raise
    except Exception as e:
        logging.error("Failed to load model from: %s", file_path)
        raise e

    
def load_data(file_path: str):
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded from: %s", file_path)
        return df 
    except pd.errors.ParserError as e:
        logging.error("Failed to parse CSV file: %s", e)
        raise 
    except Exception as e:
        logging.error("Failed to load data from: %s", e)
        raise 
    

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log metrics to MLFlow"""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logging.info("Model evaluated successfully")
        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
        logging.info("Model evaluation metrics calculated: %s", metrics_dict)
        return metrics_dict

    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise 
