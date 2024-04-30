from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

## Load the model
def load_model():
	model_path = './Trained-Models/RF_Loan_model.joblib' ## Define model path to load the model
	model = joblib.load(model_path)
	return model

## Perform parsing
def dataParser(data):
	listData=[]
	listData.append(data['Gender'])
	listData.append(data['Married'])
	listData.append(data['Dependents'])
	listData.append(data['Education'])
	listData.append(data['Self_Employed'])
	listData.append(data['LoanAmount'])
	listData.append(data['Loan_Amount_Term'])
	listData.append(data['Credit_History'])
	listData.append(data['Property_Area'])
	listData.append(data['TotalIncome'])
	return listData

## Defining data type of input and convert input to specified data format
class LoanPred(BaseModel): 
	Gender: float
	Married: float
	Dependents: float
	Education: float
	Self_Employed: float
	LoanAmount: float
	Loan_Amount_Term: float
	Credit_History: float
	Property_Area: float
	TotalIncome: float


## Defining the paths
@app.get('/') ## Default path
def index():
    return {'message': 'Welcome to Loan Prediction App'}

## Defining the function which will make the prediction using the data which the user inputs (JSON)
@app.post('/predict') ## On sending POST request to 'root/predict' with JSON input, we load the model, perform prediction & return output as JSON 
def predict_loan_status(loan_details: LoanPred): ## JSON input passed through the instructor
	model = load_model() ## Load model
	data = loan_details.model_dump() ## Converts from json to python dict
	data=dataParser(data) ## Parse data to list

	## Predictions 
	prediction = model.predict([data])

	if prediction == 0:
		pred = 'Rejected'
	else:
		pred = 'Approved'

	return {'Status of Loan Application: ':pred}

if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)