import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import src.evaluate_performance as ep
import logging

def test_evaluate_normal():

    scores = pd.DataFrame({
    "truth": [0, 1, 0, 1, 0],
    "label": [0, 1, 1, 1, 0],
    "probability": [0.1, 0.8, 0.6, 0.9, 0.3]
    })
    
    kwargs = {
        "score_type": {
            "truth": "truth",
            "label": "label",
            "probability": "probability"
        },
        "metric_type": {
            "auc": "AUC",
            "accuracy": "Accuracy",
            "confusion": "Confusion Matrix",
            "classification_report": "Classification Report"
        },
        "threshold": {
            "auc": 0.7,
            "accuracy": 0.8
        }
    }

    ep.evaluate_performance(scores,kwargs)

## Test Case 2: Invalid entry in probability
def test_invalid_prob():

    scores = pd.DataFrame({
    "truth": [0, 1, 0, 1, 0],
    "label": [0, 1, 1, 1, 0],
    "probability": ['abc', 0.8, 0.6, 0.9, 0.3]
    })
    
    kwargs = {
        "score_type": {
            "truth": "truth",
            "label": "label",
            "probability": "probability"
        },
        "metric_type": {
            "auc": "AUC",
            "accuracy": "Accuracy",
            "confusion": "Confusion Matrix",
            "classification_report": "Classification Report"
        },
        "threshold": {
            "auc": 0.7,
            "accuracy": 0.8
        }
    }
    
    with pytest.raises(ValueError):
        ep.evaluate_performance(scores,kwargs)

## Test case 3: Invalid value in truth
def test_invalid_truth():

    scores = pd.DataFrame({
    "truth": ['a', 1, 0, 1, 0],
    "label": [0, 1, 1, 1, 0],
    "probability": [0.1, 0.8, 0.6, 0.9, 0.3]
    })
    
    kwargs = {
        "score_type": {
            "truth": "truth",
            "label": "label",
            "probability": "probability"
        },
        "metric_type": {
            "auc": "AUC",
            "accuracy": "Accuracy",
            "confusion": "Confusion Matrix",
            "classification_report": "Classification Report"
        },
        "threshold": {
            "auc": 0.7,
            "accuracy": 0.8
        }
    }
    
    with pytest.raises(Exception):
        ep.evaluate_performance(scores,kwargs)


## Test Case 4: Incorrect column name

def test_invalid_colname():

    scores = pd.DataFrame({
    "truth": [1, 1, 0, 1, 0],
    "label": [0, 1, 1, 1, 0],
    "probability": [0.1, 0.8, 0.6, 0.9,0.6]
    })
    
    kwargs = {
        "score_type": {
            "truth": "truth",
            "label": "wrong_label",
            "probability": "probability"
        },
        "metric_type": {
            "auc": "AUC",
            "accuracy": "Accuracy",
            "confusion": "Confusion Matrix",
            "classification_report": "Classification Report"
        },
        "threshold": {
            "auc": 0.7,
            "accuracy": 0.8
        }
    }
    
    with pytest.raises(KeyError):
        ep.evaluate_performance(scores,kwargs)


## Test Case 5: Invalid metric
def test_invalid_metric():

    scores = pd.DataFrame({
    "truth": [1, 1, 0, 1, 0],
    "label": [0, 1, 1, 1, 0],
    "probability": [0.1, 0.8, 0.6, 0.9,0.6]
    })
    
    kwargs = {
        "score_type": {
            "truth": "truth",
            "label": "label",
            "probability": "probability"
        },
        "metric_type": {
            "auc": "AUC",
            "accuracy": "Accuracy",
            "conf_wrong": "Confusion Matrix",
            "classification_report": "Classification Report"
        },
        "threshold": {
            "auc": 0.7,
            "accuracy": 0.8
        }
    }
    
    with pytest.raises(KeyError):
        ep.evaluate_performance(scores,kwargs)

## Test Case 6: Model scored correctly
def test_correct_log(caplog):
    scores = pd.DataFrame({
    "truth": [0, 1, 0, 1, 0],
    "label": [0, 1, 1, 1, 0],
    "probability": [0.5, 0.5, 0.5, 0.5, 0.5]
    })
    
    kwargs = {
        "score_type": {
            "truth": "truth",
            "label": "label",
            "probability": "probability"
        },
        "metric_type": {
            "auc": "AUC",
            "accuracy": "Accuracy",
            "confusion": "Confusion Matrix",
            "classification_report": "Classification Report"
        },
        "threshold": {
            "auc": 0.999,
            "accuracy": 0.8
        }
    }

    metrics = ep.evaluate_performance(scores,kwargs)  

    curr_auc = metrics['metrics'][kwargs["metric_type"]["auc"]]

    assert (curr_auc < kwargs.get('threshold',None).get('auc',None))




