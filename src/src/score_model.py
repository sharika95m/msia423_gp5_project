import sys
import logging
from pathlib import Path
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def score_model(test_data: pd.DataFrame, rf_model: RandomForestClassifier,
                kwargs) -> pd.DataFrame:
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

        #Predicting probability of cloud class being 0 or 1
        ypred_proba_test = rf_model.predict_proba(test_x)[:,1]

        #Predicting label of cloud class
        ypred_bin_test = rf_model.predict(test_x)

        y_pred_dict = {kwargs["score_type"]["probability"]: ypred_proba_test,
                    kwargs["score_type"]["label"]: ypred_bin_test,
                    kwargs["score_type"]["truth"]: test_y}
        pred = pd.DataFrame(y_pred_dict)
        logger.info("Predictions and respective probabilities \
                    have been calculated.")
    except IndexError as i_err:
        logger.error("While finding the predictions, \
                    Index error has occured: %s", i_err)
        sys.exit(1)
    except ValueError as v_err:
        logger.error("While finding the predictions, \
                    Value error has occured: %s", v_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While finding the predictions, \
                    Other error has occured: %s", other)
        sys.exit(1)

    return pred

def save_scores(predictions: pd.DataFrame, save_path: Path) -> None:
    """Saves scores dataset to CSV file

    Args:
        predictions: dataframe to be stored as csv
        save_path: Local path to write csv to
    """
    try:
        predictions.to_csv(save_path)
        logger.info("Datasets written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("While saving scores, FileNotFoundError \
                    has occured: %s", fnfe)
        sys.exit(1)
    except IOError as io_err:
        logger.error("While saving scores, IO Error \
                    has occured: %s", io_err)
        sys.exit(1)
    except Exception as other:
        logger.error("Error occurred while trying to \
                    write dataset to file: %s", other)
        sys.exit(1)
