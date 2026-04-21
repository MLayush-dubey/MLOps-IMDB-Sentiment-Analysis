import numpy as np 
import pandas as pd 
import os 
import re 
import nltk 
from nltk.stem import WordNetLemmatizer 
import string 
import logging 
from src.logger import logging 
from nltk.corpus import stopwords 
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_dataframe(df: pd.DataFrame, col: str = 'text') -> pd.DataFrame:
    """
    Preprocess a dataframe by applying text cleaning operations.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    #initialize lemmatizer and stopwords 
    lemmatizer = WordNetLemmatizer() 
    stop_words = set(stopwords.words("english"))


    def preprocess_text(text):
        """Helper functions to preprocess a single text"""

        #convert to lowercase 
        text = text.lower()

        #remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        #remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()

        #remove stop words 
        text = ''.join([word for word in text.split() if word not in stop_words])

        #lemmatize
        text = ''.join([lemmatizer.lemmatize(word) for word in text.split()])

        #remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        return text 

    
    #apply preprocessing to the specified column 
    df[col] = df[col].apply(preprocess_text)

    #remove small sentences(less than 3 words)
    df[col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 3]))

    #drop rows with NaN values 
    df = df.dropna(subset = [col])
    logging.info("Data pre-processing completed")
    return df 

    

def main():
    try:
        #fetch data from data/raw
        train_data = pd.read_csv("C:\\Users\\PC\\Documents\\MLOps-IMDB-Sentiment-Analysis\\data\\raw\\train.csv")
        test_data = pd.read_csv("C:\\Users\\PC\\Documents\\MLOps-IMDB-Sentiment-Analysis\\data\\raw\\test.csv")
        logging.info("data loaded properly")

        #transform the data 
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_preprocessed_data = preprocess_dataframe(test_data, 'review')

        #store the data inside 
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok = True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index = False)  #basically interim folder meh as a csv train preprocessed data store kr rahe hai
        test_preprocessed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index = False) 

        logging.info("Processed data saved to %s", data_path) 
    except Exception as e:
        logging.error("Failed to complete the data transformation process: %s", str(e))
        print(f"Error: {e}")


if __name__ == "__main__":
    main()



    
