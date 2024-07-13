# Project Documentation: Medical Cost Prediction

## Table of Contents

### Introduction

### Dataset Description

### Project Objectives

### Project Structure

### Data Ingestion

### Data Transformation

### Model Training

### Training Pipeline

### Prediction Pipeline

### Flask

### Logging

### Exception Handling

### Utils

### Logging

### Conclusion



## 1. Introduction


The project aims to predict medical insurance costs based on features such as age, BMI, and smoking status using the "Medical Cost Personal Dataset" from Kaggle. This document provides a comprehensive overview of the project, including its structure, processes, and supporting scripts.



## 2. Dataset Description


Dataset Name : Medical Cost Personal Dataset

Website : Kaggle

Link : Medical Cost Dataset



## Description


The dataset contains information on the medical costs billed by health insurance. The features include:


Age: Age of the individual.

Sex: Gender of the individual (male, female).

BMI: Body Mass Index, a measure of body fat based on height and weight.

Children: Number of children/dependents covered by health insurance.

Smoker: Whether the individual smokes (yes, no).

Region: The individual's residential area in the US (northeast, southeast, southwest, northwest).

Charges: The medical insurance cost billed to the individual.



## 3. Project Objectives


Data Ingestion: Load and explore the dataset.

Data Transformation: Clean, preprocess, and transform the dataset for modeling.

Model Training: Train various regression models to predict medical insurance costs.

Pipeline Creation: Develop a pipeline for data transformation, model training, and prediction.

Supporting Scripts: Provide scripts for setup, exception handling, utilities, and logging.


## 4. Project Structure

```
MedicalCostPrediction/
│
├── artifacts/
│   ├── (best)model.pkl
│   ├── linearRegression.pkl
│   ├── Lasso.pkl
│   ├── Ridge.pkl
│   ├── ElasticNet.pkl
│   ├── DescisionTreeRegressor.pkl
│   ├── RandomForestRegressor.pkl
│   ├── GradientBoostingRegressor.pkl
│   ├── AdaBoostRegressor.pkl
│   ├── XGBoostRegressor.pkl
│   ├── raw.csv
│   └── preprocessor.pkl
│
│
├── notebooks/
│   ├── med.jpeg
│   ├── insurance.csv
│   ├── model_evaluaion(chart).jpg
│   └── Medical_Cost_Prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── components /
│   │           ├── __init__.py
│   │           ├── data_ingestion.py
│   │           ├── data_transformation.py
│   │           └──  model_training.py
│   │
│   ├── pipeline / 
│   │           ├── __init__.py
│   │           ├── training_pipeline.py
│   │           └── prediction_pipeline.py
│   │ 
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│     
├── templates / 
│     ├── index.html
│     └── results.html
│   
├──static / 
│      ├── med.jpeg
│      └── style.css               
│
├── app.py
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py


```

## 5. Data Ingestion

The data ingestion file is used to ingest the data from notebook/data and split it into three parts: train.csv, test.csv, and raw.csv, and save them into the artifacts folder.


## 6. Data Transformation

The data transformation file is used to perform EDA (encoding, preprocessing the data) and save the encoded data.


## 7. Model Training

The model training file is used to train the model with various regression algorithms, save the best model in pkl file format in the artifacts folder, and save all models.


## 8. Training Pipeline

This file is used to run data ingestion, data transformation, and model training scripts.


## 9. Prediction Pipeline

This file is used to predict the data using the best_model.pkl file and preprocess the data from preprocessor.pkl.


## 10. Static

static/style.css: This file provides the theme to the index.html and results.html.


## 11. Templates

templates/index.html: This file is used to create a form to get data from the user.

templates/results.html: This file is used to show the predicted results of the model.


## 12. Flask (app.py)

This file is used to post data from the form (index.html), predict the results using the prediction_pipeline.py, and show the results.


## 13. Logger

This file is used to save the logs. Logging is implemented to record the execution flow and errors.


## 14. Exception

This file is used to write the exception handling code for errors. Exception handling is implemented to ensure that errors are caught and logged appropriately.


## 15. Utils

This file is used to write functions used in the code. Utility functions are provided for common tasks such as creating directories.


## 16. Conclusion

This documentation provides a comprehensive overview of the "Medical Cost Personal Dataset" project, covering data ingestion, transformation, model training, pipeline creation, and supporting scripts. The provided code examples illustrate the implementation of various components, ensuring a robust and scalable project structure.






