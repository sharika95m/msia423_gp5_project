from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('Modeling/best_model.pkl')

# Define a route for the API
@app.route('/predict', methods=['POST'])

def predict():
    # Get the input data as a JSON object
    input_data = request.get_json(force=True)

    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Make predictions using the loaded model
    prediction = model.predict(input_df)

    # Return the prediction as a JSON object
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
