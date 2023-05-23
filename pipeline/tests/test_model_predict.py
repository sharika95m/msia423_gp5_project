import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pytest
from pathlib import Path
import src.model_predict as mp

kwargs = {
    "choose_features": {"features_to_use": ["feature1", "feature2"]},
    "get_target": {"target_feature": "target"},
    "score_type": {"probability": "prob", "label": "label", "truth": "truth"}
}

model = DecisionTreeClassifier()

test_data = pd.DataFrame({
    "feature1": [1, 2, 3],
    "feature2": [4, 5, 6],
    "target": [0, 1, 0]
})

    # Fit the model
model.fit(test_data[kwargs["choose_features"]["features_to_use"]], test_data[kwargs["get_target"]["target_feature"]])


# Test case 1: Valid input
def test_model_predict_valid_input():
    # Create test data
    test_data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "target": [0, 1, 0]
    })
    
    kwargs = {
        "choose_features": {"features_to_use": ["feature1", "feature2"]},
        "get_target": {"target_feature": "target"},
        "score_type": {"probability": "prob", "label": "label", "truth": "truth"}
    }
    
    # Call model_predict function
    result_data, prediction_df = mp.model_predict(test_data, model, kwargs)
    
    # Assert the shape of result_data and prediction_df
    assert result_data.shape == (3, 5)
    assert prediction_df.shape == (3, 3)
    
# Test case 2: Empty test data
def test_model_predict_empty_test_data(caplog):
  
    # Define kwargs
    kwargs = {
        "choose_features": {"features_to_use": ["feature1", "feature2"]},
        "get_target": {"target_feature": "target"},
        "score_type": {"probability": "prob", "label": "label", "truth": "truth"}
    }
    
    ## Empty Data set
    test_data = pd.DataFrame(columns=["feature1", "feature2", "target"])

    # Call model_predict function
    with pytest.raises(SystemExit):
        result_data, prediction_df = mp.model_predict(test_data, model, kwargs)


# Test case 3: Error handling - IndexError
def test_model_predict_index_error():
    # Create test data with missing columns
    test_data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "target": [0, 1, 0]
    })

    # Define kwargs with incorrect feature name
    kwargs = {
        "choose_features": {"features_to_use": ["feature1", "feature2"]},
        "get_target": {"target_feature": "target"},
        "score_type": {"probability": "prob", "label": "label", "truth": "truth"}
    }
    
    # Call model_predict function and expect it to raise an IndexError
    with pytest.raises(SystemExit):
        result_data, prediction_df = mp.model_predict(test_data, model, kwargs)


# Test case 4: Error handling - TypeError
def test_model_predict_type_error():
    # Create test data with invalid data type
    test_data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 'abv'],  # Invalid data type
        "target": [0, 1, 0]
    })
    

    
    # Define kwargs
    kwargs = {
        "choose_features": {"features_to_use": ["feature1", "feature2"]},
        "get_target": {"target_feature": "target"},
        "score_type": {"probability": "prob", "label": "label", "truth": "truth"}
    }
    
    # Call model_predict function and expect it to raise a TypeError
    with pytest.raises(SystemExit):
        result_data, prediction_df = mp.model_predict(test_data, model, kwargs)

## Test case 5: Missing file path

def test_save_predict():
    test_data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "target": [0, 1, 0]
    }) 

    prediction_df = pd.DataFrame({kwargs["score_type"]["truth"]: [0,1,0],
                kwargs["score_type"]["probability"]: [0.1,0.9,0.1],
                kwargs["score_type"]["label"]: [0,1,0]})
    
    save_path = Path('/random_path/check.csv')

    with pytest.raises(SystemExit):
        mp.save_predictions(test_data,prediction_df,save_path)

