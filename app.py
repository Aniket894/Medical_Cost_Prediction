from flask import Flask, request, render_template, jsonify
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

# Import the PredictPipeline and CustomData classes
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

application = Flask(__name__, template_folder="templates")
app = application

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Get data from form
        age = int(request.form.get("age"))
        sex = request.form.get("sex")
        bmi = float(request.form.get("bmi"))
        children = int(request.form.get("children"))
        smoker = request.form.get("smoker")
        region = request.form.get("region")
        
        # Prepare the input data using CustomData class
        custom_data = CustomData(age, sex, bmi, children, smoker, region)
        input_df = custom_data.get_data_as_dataframe()
        
        # Predict using the PredictPipeline
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)
        
        # Prepare the result to be rendered
        final_result = prediction[0]
        
        return render_template("results.html", final_result=final_result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
