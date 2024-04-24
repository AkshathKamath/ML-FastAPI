1. The Random Forest Classifier model is trained on the 'train.csv' & the trained model is then saved in the 'Trained-Models' folder by running the 'loan_prediction' python file.
2. Run the 'model_api' python file, which loads the trained model and creates a REST API which listens to POST requests on 'localhost:8000/predict' with JSON body as input for model and returns JSON response with the prediction from the model.

Sample JSON input data:  
{  
"Gender": 0,  
"Married": 0,  
"Dependents": 0,  
"Education": 0,  
"Self_Employed": 0,  
"LoanAmount": 4.9875,  
"Loan_Amount_Term": 360.0,  
"Credit_History": 1,  
"Property_Area": 2,  
"TotalIncome": 8.698  
}
