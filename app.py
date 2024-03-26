from flask import Flask, request, render_template
import numpy as np 
import pandas as pd

from sklearn.preprocessing import StandardScaler 
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

# Create a flask application 
application = Flask(__name__)
app = application

# Route for homepage 
@app.route("/")

def index():
    return render_template("index.html")

# Route for making predictions 
@app.route("/predictdata", methods = ["GET", "POST"])
# Predict 

def predict_data_point():
    if request.method == "GET": # If request is GET render home.html template
        return render_template("home.html")
    else:
        # Initialize an empty dictionary to store the data 
        form_data = {}

        # Iterate over form fields and populate the dictionary 
        for field in ['Longitude', 'Latitude', 'Housing Median Age', 'Total Rooms', 'Total Bedrooms',
                                  'Population', 'Households', 'Median Income', 'Ocean Proximity']:
            form_data[field] = request.form.get(field)

        # Create custom data object using kwargs
        custom_data = CustomData(**form_data)

        # Convert the form_data dictionary to  a DataFrame
        pred_df = custom_data.get_data_as_dataframe()

        # Print the DataFrame for debugging process 
        print(pred_df)

        # Log the start of the prediction process 
        print("Before Prediction")

        # Initialize predict pipeline 
        predict_pipeline = PredictionPipeline()

        # Log the midpoint of the prediction process 
        print("Mid Prediction")

        # Make predictions 
        results = predict_pipeline.make_predictions(pred_df)

        # Log the end of the prediction process 
        print("After Prediction")

        # Return the results 
        return render_template("home.html", results = results[0])
    

# Run the Flask app
if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug=True)