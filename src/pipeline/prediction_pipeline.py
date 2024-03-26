import sys 
import pandas as pd 
import os 

from src.exception import FileOperationError
from src.utils import load_object 


# Create a class for making predictions using the trained model 
class PredictionPipeline:
    def __init__(self):
        pass 


    def make_predictions(self, features):
        try:
            # Define paths for the model and preprocessor 
            model_path = os.path.join("data_repository", "model.joblib")
            preprocessor_path = os.path.join("data_repository", "preprocessor.joblib")

            # Load the trained model and preprocessor 
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            # Transform input features using preprocessor
            scaled_data = preprocessor.transform(features)

            # Make predictions using the model 
            prediction = model.predict(scaled_data)

            # Return predictions 
            return prediction
        
        except Exception as e:
            raise FileOperationError(e, sys)
        
# Create a dataclass to represent input features/data for prediction 
# We create a class that allows us to create instances representing individual points 
class CustomData:
    def __init__(self, **kwargs):
        # Initialize attributes using **kwargs 
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Define a method to convert data object to a dataframe 
    def get_data_as_dataframe(self):
        try:
            # Create a dictionary from custom data attributes 
            data_dict = {key: [getattr(self, key)] for key in vars(self)} 

            # Convert the dictionary to a Dataframe
            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise FileOperationError(e, sys)
        
