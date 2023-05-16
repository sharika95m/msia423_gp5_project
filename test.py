import pandas as pd
import requests
import json
from sklearn.metrics import *



# Read test data
test = pd.read_csv('test.csv')

# Split into test x and y
test_x = test.drop(columns = ['Churn_Yes', 'Unnamed: 0'])
test_y = test['Churn_Yes']

# URL of flask app
api_url = 'http://0.0.0.0:5000/predict'

# Num rows to test
num_test_rows = len(test_y)

# Loop through the validation dataset and send requests to the API
results = []
for idx, row in test_x.iloc[0:num_test_rows, :].iterrows():
    # Convert the row to a dictionary
    data = [row.to_dict()]
    
    # Send a POST request to the API with the data
    response = requests.post(api_url, json=data)
    
    # Parse the JSON response and store the prediction
    prediction = json.loads(response.text)
    results.append(prediction)

# Convert the results to a DataFrame
results_df = pd.DataFrame(results, columns=['prediction'])

# Metrics
print('Test F1 score: ', f1_score(test_y, results_df['prediction']))
print('Test Accuracy Score: ', accuracy_score(test_y, results_df['prediction']))

# Save the results to a CSV file (optional)
results_df.to_csv('api_results.csv', index=False)

# Print the results
print(results_df)