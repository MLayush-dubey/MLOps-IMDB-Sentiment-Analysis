import os 
import pickle 

#define the path to the .pkl file 
pkl_path = "../models/vectorizer.pkl"

#check if the file path exists 
if os.path.exists(pkl_path):
    print(f"File found: {pkl_path}")
    try:
        #attempt to load the file 
        vectorizer = pickle.load(open(pkl_path, "rb")) 
        print("File loaded successfully") 
    except Exception as e:
        print(f"Error loading the .pkl file: {e}")
    
else:
    print(f"File not found: {pkl_path}. Please check the path and try again")