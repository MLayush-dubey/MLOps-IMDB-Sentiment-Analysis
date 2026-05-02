#Script to promote the model if better than previous versions

import os 
import mlflow 


def promote_model():
    #Setup dagshub cred for MLFlow tracking 
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN env variable not set")

    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token 
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "MLayush-dubey"
    repo_name = "MLOps-IMDB-Sentiment-Analysis"

    #Setup MLFLow tracking URI
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    client = mlflow.MLflowClient() 

    model_name = "my_model"

    #Get the latest version in staging 
    latest_version_staging = client.get_latest_versions(model_name, stages = ["Staging"])[0].version 
    #Returns a list of ModelVersion objects and grabs the latest model

    #Archive the current production model 
    prod_versions = client.get_latest_versions(model_name, stages = ["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name = model_name,
            version = version.version,
            stage = "Archived"
        )

    #Promote the model to production 
    client.transition_model_version_stage(
        name = model_name, 
        version = latest_version_staging,
        stage = "Production"
    )

    print(f"Model Version {latest_version_staging} promoted to Production")
    
if __name__ == "__main__":
    promote_model()
        