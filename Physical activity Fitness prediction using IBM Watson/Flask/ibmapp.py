import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
import requests
import json

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "bHeQAmmMoAl8xTfHXY7ncnJqgNYD1Aa95yvNQMxunTdn"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
model = pickle.load(open('../model building/fitness.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/home1')
def home1():
    return render_template('home.html')
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['sad','neutral','happy','step_count',
                     'calories_burned','hours_of_sleep','weight_kg']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    print(output)

    payload_scoring = {"input_data": [{"field": ['sad', 'neutral', 'happy', 'step_count',
                                                 'calories_burned', 'hours_of_sleep', 'weight_kg'],
                                       "values": [input_features]}]}

    response_scoring = requests.post(
        'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/8e148d11-ed02-4316-805a-fef7920d12a6/predictions?version=2022-07-27',
        json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    prediction=response_scoring.json()
    print(prediction)
    output=prediction["predictions"][0]["values"][0][0]


    if (output==0):
        return render_template("result1.html")
    else:
        return render_template("result.html")
if __name__=='__main__':
    app.run(debug=False)