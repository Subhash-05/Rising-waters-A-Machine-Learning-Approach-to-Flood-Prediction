from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():

    try:
        # Get values from form
        cloud = float(request.form["Cloud Cover"])
        annual = float(request.form["ANNUAL"])
        janfeb = float(request.form["Jan-Feb"])
        marmay = float(request.form["Mar-May"])
        junsep = float(request.form["Jun-Sep"])

        # IMPORTANT:
        # Your model was trained on 10 features.
        # So we must supply all 10 features in correct order.
        # For missing ones (Temp, Humidity, Oct-Dec, avgjune, sub),
        # we can use average values from dataset.

        # Replace these with dataset mean values if needed
        temp = 29
        humidity = 73
        octdec = 500
        avgjune = 250
        sub = 500

        input_data = pd.DataFrame([[temp, humidity, cloud,
                                    annual, janfeb, marmay,
                                    junsep, octdec,
                                    avgjune, sub]],
                                  columns=["Temp","Humidity","Cloud Cover",
                                           "ANNUAL","Jan-Feb","Mar-May",
                                           "Jun-Sep","Oct-Dec",
                                           "avgjune","sub"])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            return render_template("chance.html")
        else:
            return render_template("no chance.html")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
