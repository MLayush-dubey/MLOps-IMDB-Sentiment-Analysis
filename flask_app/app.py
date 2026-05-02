from flask import Flask, render_template, request 
import mlflow 
import pickle 
import os 
import pandas as pd 
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST 
import time 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
import string 
import dagshub 
import re 

import warnings 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


def lemmatization(text):
    """Lemmatize the text""" 
    lemmatizer = WordNetLemmatizer() 
    text = text.split() 
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text) 


def remove_stop_words(text):
    """Remove stop words from the text""" 
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    """Remove numbers from the text""" 
    text = [char for char in str(text) if not char.isdigit()]
    return "".join(text)


def lower_case(text):
    """Convert text to lowercase"""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text


def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    """Remove sentences with less than 3 words"""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan 


def normalize_text(text):
    """Normalize the text""" 
    text = lower_case(text)
    text = removing_numbers(text) 
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = remove_small_sentences(text)
    text = remove_stop_words(text)
    text = lemmatization(text)
    return text


#Below code block is for local use
# mlflow.set_tracking_uri('https://dagshub.com/MLayush-dubey/MLOps-IMDB-Sentiment-Analysis.mlflow')
# dagshub.init(repo_owner = "MLayush-dubey", repo_name = "MLOps-IMDB-Sentiment-Analysis")

#-------------------------------------------------------------------------------------------------
#Below code is for production use
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token: 
    raise EnvironmentError("DAGSHUB_TOKEN env variable not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token 
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token 

dagshub_url = "https://dagshub.com"
repo_owner = "MLayush-dubey"
repo_name = "MLOps-IMDB-Sentiment-Analysis" 

#Setup MLFlow tracking URI 
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

#-----------------------------------------------------------------------------------------------

#Initialize flask app 
app = Flask(__name__)


#Create custom registry--> a container that stores all your custom metrics 
registry = CollectorRegistry() 
#by default prometheus scrapes system metrics(CPU, memory, etc). But since we want custom metrics for the ML service, hence we do this


#Define your custom metrics using this registry 
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry = registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of request in seconds", ["endpoint"], registry = registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Total number of predictions per class", ["prediction"], registry = registry
)



#------------------------------------------------------------------------------------------------------
#Model and vectorizer setup 
model_name = "my_model"
def get_latest_model_version(model_name):
    """Fetches the latest approved model from MLFLow model registry"""
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages = ["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name)
    return latest_version[0].version if latest_version else None 

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)   #loads the ml model as a python function
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))   #loads the vectorizer(text preprocessing object)


#Routes 
@app.route("/")
def home():
    REQUEST_COUNT.labels(method = "GET", endpoint = "/").inc()  #inc() increment count by 1
    start_time = time.time() 
    response = render_template("index.html", result = None)
    REQUEST_LATENCY.labels(endpoint = "/").observe(time.time() - start_time)
    return response 


@app.route("/predict", methods = ["POST"])
def predict():
    REQUEST_COUNT.labels(method = "POST", endpoint = "/predict").inc() 
    start_time = time.time() 

    text = request.form['text']   #retrieves the text input from the form

    #clean text 
    text = normalize_text(text) 

    #convert to features 
    features = vectorizer.transform([text]) 
    features_df = pd.DataFrame(features.toarray(), columns = [str(i) for i in range(features.shape[1])])

    #make prediction 
    result = model.predict(features_df[0])
    prediction = result[0]

    #Increment prediction counter 
    PREDICTION_COUNT.labels(prediction = str(prediction)).inc() 

    #measure latency 
    REQUEST_LATENCY.labels(endpoint = "/predict").observe(time.time() - start_time)

    return render_template("index.html", result = prediction)


#prometheus scrapes this endpoint every X(15-default) seconds
@app.route("/metrics")
def metrics():    
    """Expose only custom prometheus metrics"""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000, debug = True)