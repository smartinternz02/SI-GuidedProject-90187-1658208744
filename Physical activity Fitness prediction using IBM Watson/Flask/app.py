import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
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
    print(features_value)
    features_name = ['sad','neutral','happy','step_count',
                     'calories_burned','hours_of_sleep','weight_kg']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    print(output)
    if (output==0):
        return render_template("result1.html")
    else:
        return render_template("result.html")
if __name__=='__main__':
    app.run(debug=False)