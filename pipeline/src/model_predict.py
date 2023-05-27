"""
Module contains functionalities to make predictions and store them
"""

import sys
import logging
from pathlib import Path
import warnings
from typing import Union, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def model_predict(test_data: pd.DataFrame,
                  model: Union[RandomForestClassifier, DecisionTreeClassifier],
                  kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summary: Checks score (prediction probabilities and
    labels) from model on test dataset.
    Function will stop execution if predictions are not made.
    Args:
        test_data: dataframe to store test data
        rf_model: Model object used for scoring
        kwargs: Dictionary with feature names and target label
    """
    try:
        #Extracting X and y for test data
        test_x = test_data[kwargs["choose_features"]["features_to_use"]]
        test_y = test_data[kwargs["get_target"]["target_feature"]]

        #Predicting probability of churn
        predict_probability = model.predict_proba(test_x)[:,1]
        test_data[kwargs["score_type"]["probability"]] = predict_probability

        #Predicting 0 (no churn) or 1 (churn)
        predictions = model.predict(test_x)
        test_data[kwargs["score_type"]["label"]] = predictions


        prediction_df = pd.DataFrame({kwargs["score_type"]["truth"]: test_y,
                        kwargs["score_type"]["probability"]: predict_probability,
                        kwargs["score_type"]["label"]: predictions})

        logger.info("Predictions and respective probabilities \
                    have been calculated.")
    except IndexError as i_err:
        print(i_err)
        logger.error("Index error occurred while finding predictions: %s", i_err)
        sys.exit(1)
    except ValueError as v_err:
        print(v_err)
        logger.error("Value error occurred while finding predictions: %s", v_err)
        sys.exit(1)
    except TypeError as t_err:
        print(t_err)
        logger.error('Type error occured while finding predictions,\
                      possible type mismatch between data type: %s', t_err)
        sys.exit(1)
    except Exception as other:
        print(other)
        logger.error("Error occured While finding the predictions: %s", other)
        sys.exit(1)

    return test_data, prediction_df

def save_predictions(data: pd.DataFrame, predictions: pd.DataFrame, save_path: Path) -> None:
    """
    Summary: Saves dataset with predictions to CSV file
    Args:
        data: dataframe to be stored as csv
        save_path: Local path to write csv to
    """
    try:
        data.to_csv(save_path / "data_with_predictions.csv", index=False)
        predictions.to_csv(save_path / "predictions.csv", index=False)
        logger.info("Datasets written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("FileNotFoundError occured While saving scores: %s", fnfe)
        sys.exit(1)
    except IOError as io_err:
        logger.error("IO Error occured While saving scores,\
                     possible incorrect file path: %s", io_err)
        sys.exit(1)
    except Exception as other:
        logger.error("Error occurred while trying to \
                    write dataset to file: %s", other)
        sys.exit(1)
