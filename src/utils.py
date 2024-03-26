import os 
import sys 
import joblib
#import dill 
#import pickle 
from sklearn.metrics import r2_score

from src.exception import FileOperationError
from src.log_config import logging 

# Define a function to save an object in a file 
def save_object(file_path, obj):
    try:
        # Extract the directory path from the given file path. 
        
        # Get the directory path of the file 
        dir_path = os.path.dirname(file_path)
        # if the directory path does not exist, create it 
        os.makedirs(dir_path, exist_ok = True)

        # save the object in the file 
        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)

    # Handle exception 
    except Exception as e:
        raise FileOperationError(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, best_model):
    try:
        # Log message for evaluation 
        logging.info("Evaluating the best model")

        # Fit the best model to training data
        best_model.fit(X_train, y_train)

        # Predict the target variable for training and testing data 
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Calculate the R-squared score for training and testing data 
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Create a report containing the R-squared scores
        report = f"R-squared score for training data: {r2_train}\n R-squared score for testing data: {r2_test}"
        logging.info(report)

        # Return the R-squared scores 
        return report
    
    # Handle exception
    except Exception as e:
        raise FileOperationError(e, sys) 
    

def load_object(file_path):
    try:
        # Open the file in a binary read mode and load the object using joblib 
        # Load the object from file using joblib's load function 
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_path)
        
    # Handle exception
    except Exception as e:
        raise FileOperationError(e, sys)
    

