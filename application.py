import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler



application=Flask(__name__)
app=application

##import ridge regressorand standardscaler pickle
rigde_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Get form inputs and convert them to float
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            Ws = float(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            ISI = float(request.form['ISI'])
            Classes = float(request.form['Classes'])
            Region = float(request.form['Region'])

            # Prepare input data for prediction
            features = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

            # Standardize the features using the scaler
            scaled_features = standard_scaler.transform(features)

            # Predict the result using the Ridge model
            prediction = rigde_model.predict(scaled_features)

            # Return the result to the template
            return render_template('home.html', result=prediction[0])

        except Exception as e:
            # Print the error in the terminal for debugging
            print(f"Error: {e}")
            # If there's an error, return an appropriate message
            return render_template('home.html', result="An error occurred. Please check your inputs.")

    # If the method is GET, render the form
    return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")