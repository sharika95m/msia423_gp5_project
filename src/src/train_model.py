import sys
import warnings
import logging
from pathlib import Path
from typing import Tuple
import pickle
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def train_model(data: pd.DataFrame, kwargs) -> \
    Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Summary: train_model function performs 3 key operations
    1. Create train and test dataset
    2. Build model on train data
    3. Save model and train and test datasets to required directory
    Args:
        data: dataframe to perform modeling on
        kwargs: Dictionary with information for train-test split
        and model hyperparameters
    """
    try:
        train, test = train_test_split(data, **kwargs["split_data"])

        train_x = train[kwargs["choose_features"]["features_to_use"]]
        train_y = train[kwargs["get_target"]["target_feature"]]

        model = RandomForestClassifier(**kwargs["model_training"])
        model.fit(train_x, train_y)

        logger.info("Model and train-test datasets have been created.")
    except KeyError as k_err:
        logger.error("During train-test split and/or modeling, \
                    Key error has occured: %s", k_err)
        sys.exit(1)
    except AttributeError as a_err:
        logger.error("During train-test split and/or modeling, \
                    Attribute error has occured: %s", a_err)
        sys.exit(1)
    except ValueError as v_err:
        logger.error("During train-test split and/or modeling, \
                    Value error has occured: %s", v_err)
        sys.exit(1)
    except Exception as other:
        logger.error("During train-test split and/or modeling, \
                    Other error has occured: %s", other)
        sys.exit(1)

    return model, train, test

def save_data(train_data: pd.DataFrame, test_data:
            pd.DataFrame, save_path: Path) -> None:
    """
    Summary: Saves train and test dataset to CSV file
    Function will stop executing if save_data fails
    since test dataset is needed further in pipeline.
    Args:
        train_data, test_data: dataframe to be stored as csv
        save_path: Local path to write csv to
    """
    try:
        train_data.to_csv(save_path / "train.csv")
        test_data.to_csv(save_path / "test.csv")
        logger.info("Datasets written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("While saving train/test dataset, \
                    File Not Found Error has occured: %s", fnfe)
        sys.exit(1)
    except IOError as io_err:
        logger.error("While saving train/test dataset, \
                    IO Error has occured: %s", io_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While saving train/test dataset, \
                    Other Error has occured: %s", other)
        sys.exit(1)

def save_model(rf_model: sklearn.ensemble._forest.RandomForestClassifier,
                save_path: Path) -> None:
    """
    Summary: Saves RandomForestClassifier to model file
    Function will stop executing if save_model fails
    since model is needed further in pipeline.
    Args:
        rf_model: model to store as binary file
        save_path: Local path to write pickle to
    """
    try:
        pickle.dump(rf_model, save_path.open("wb"))
        logger.info("Model written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("While saving model, \
                    File Not Found Error has occured: %s", fnfe)
        sys.exit(1)
    except IOError as io_err:
        logger.error("While saving model, \
                    IO Error has occured: %s", io_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While saving model, \
                    Other Error has occured: %s", other)
        sys.exit(1)
