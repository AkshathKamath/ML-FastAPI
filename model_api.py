from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

def load_model():
	model_path = './Trained-Models/RF_Loan_model.joblib' ## Define model path to load the model
	model = joblib.load(model_path)
	return model


## Perform parsing
class LoanPred(BaseModel): ## Defining data type of input and convert input to specified data format
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
	model = load_model()
	data = loan_details.model_dump() ## Converts from json to python dict
	gender = data['Gender']
	married = data['Married']
	dependents = data['Dependents']
	education = data['Education']
	self_employed = data['Self_Employed']
	loan_amt = data['LoanAmount']
	loan_term = data['Loan_Amount_Term']
	credit_hist = data['Credit_History']
	property_area = data['Property_Area']
	income = data['TotalIncome']

	## Predictions 
	prediction = model.predict([[gender, married, dependents, education, self_employed, loan_amt, loan_term, credit_hist, property_area,income]])

	if prediction == 0:
		pred = 'Rejected'
	else:
		pred = 'Approved'

	return {'Status of Loan Application: ':pred}

if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)