"""
Summary: Conduct cross validation
"""
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, Union, List
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def define_folds(kwargs: Dict) -> KFold:
    """
    Summary: Function to create folds for cross-validation
    Args:
        kwargs: Dictionary with information for
        creating 10-fold cross-validation
    """
    try:
        folds = KFold(**kwargs)
        logger.info("The folds for cross-validation are created.")
    except KeyError as k_err:
        logger.error(
            "While making folds for cross-validation, \
                    Key error has occured: %s",
            k_err,
        )
        sys.exit(1)
    except AttributeError as a_err:
        logger.error(
            "While making folds for cross-validation, \
                    Attribute error has occured: %s",
            a_err,
        )
        sys.exit(1)
    except ValueError as v_err:
        logger.error(
            "While making folds for cross-validation, \
                    Value error has occured: %s",
            v_err,
        )
        sys.exit(1)
    except Exception as other:
        logger.error(
            "While making folds for cross-validation, \
                    Other error has occured: %s",
            other,
        )
        sys.exit(1)

    return folds


def get_hyperparameters(
    kwargs: Dict,
) -> Union[Dict[str, List[int]], Dict[str, List[int]], Dict[str, make_scorer]]:
    """
    Summary: Function to create folds for cross-validation
    Args:
        kwargs: Dictionary with information for
        creating 10-fold cross-validation
    """
    try:
        dt_params = kwargs["DecisionTree"]["hyperparameters"]
        rf_params = kwargs["RandomForest"]["hyperparameters"]
        logger.info("The hyperparameters are extracted from config.")
    except KeyError as k_err:
        logger.error(
            "While getting hyperparameters, \
                    Key error has occured: %s",
            k_err,
        )
        sys.exit(1)
    except AttributeError as a_err:
        logger.error(
            "While getting hyperparameters, \
                    Attribute error has occured: %s",
            a_err,
        )
        sys.exit(1)
    except ValueError as v_err:
        logger.error(
            "While getting hyperparameters, \
                    Value error has occured: %s",
            v_err,
        )
        sys.exit(1)
    except Exception as other:
        logger.error(
            "While getting hyperparameters, \
                    Other error has occured: %s",
            other,
        )
        sys.exit(1)

    return dt_params, rf_params


def gridsearchcv_dt(
    dt_params: Dict, kwargs: Dict, folds: KFold, train: pd.DataFrame
) -> KFold:
    """
    Summary: Function to perform GridSearchCV for RandomForestClassifier.
    Args:
        - dt_params: Dictionary of hyperparameters
        - kwargs: Dictionary with information for
        performing GridSearchCV
        - folds: Fold definition for doing cross-validation
        - train: Dataset to do cross-validation over
    """
    try:
        dt_model = DecisionTreeClassifier()
        search = GridSearchCV(
            dt_model, dt_params, scoring=make_scorer(f1_score), n_jobs=-1, cv=folds
        )
        train_x = train[kwargs["get_features"]["features_to_use"]]
        train_y = train[kwargs["get_target"]["target_feature"]]
        results = search.fit(train_x, train_y)
        logger.info("GridSearchCV completed for DecisionTreeClassifier.")
    except KeyError as k_err:
        logger.error(
            "While doing gridsearchcv for decision tree, \
                    Key error has occured: %s",
            k_err,
        )
        sys.exit(1)
    except AttributeError as a_err:
        logger.error(
            "While doing gridsearchcv for decision tree, \
                    Attribute error has occured: %s",
            a_err,
        )
        sys.exit(1)
    except ValueError as v_err:
        logger.error(
            "While doing gridsearchcv for decision tree, \
                    Value error has occured: %s",
            v_err,
        )
        sys.exit(1)
    except Exception as other:
        logger.error(
            "While doing gridsearchcv for decision tree, \
                    Other error has occured: %s",
            other,
        )
        sys.exit(1)

    return results


def gridsearchcv_rf(
    rf_params: Dict, kwargs: Dict, folds: KFold, train: pd.DataFrame
) -> KFold:
    """
    Summary: Function to perform GridSearchCV for RandomForestClassifier.
    Args:
        - rf_params: Dictionary of hyperparameters
        - kwargs: Dictionary with information for
        performing GridSearchCV
        - folds: Fold definition for doing cross-validation
        - train: Dataset to do cross-validation over
    """
    try:
        rf_model = RandomForestClassifier()
        search = GridSearchCV(
            rf_model, rf_params, scoring=make_scorer(f1_score), n_jobs=-1, cv=folds
        )
        train_x = train[kwargs["get_features"]["features_to_use"]]
        train_y = train[kwargs["get_target"]["target_feature"]]
        results = search.fit(train_x, train_y)
        logger.info("GridSearchCV completed for RandomForestClassifier.")
    except KeyError as k_err:
        logger.error(
            "While doing gridsearchcv for random forest, \
                    Key error has occured: %s",
            k_err,
        )
        sys.exit(1)
    except AttributeError as a_err:
        logger.error(
            "While doing gridsearchcv for random forest, \
                    Attribute error has occured: %s",
            a_err,
        )
        sys.exit(1)
    except ValueError as v_err:
        logger.error(
            "While doing gridsearchcv for random forest, \
                    Value error has occured: %s",
            v_err,
        )
        sys.exit(1)
    except Exception as other:
        logger.error(
            "While doing gridsearchcv for random forest, \
                    Other error has occured: %s",
            other,
        )
        sys.exit(1)

    return results


def train_model_dt(
    hyperparameters: Dict[str, int], train: pd.DataFrame, kwargs: Dict[str, List[str]]
) -> DecisionTreeClassifier:
    """
    Summary: Trains a DecisionTree model
    Function will stop executing if model training fails
    since it is needed further in pipeline.
    Args:
        hyperparameter: Hyperparameters for model training
    """
    try:
        # Finding X and y in train data
        train_x = train[kwargs["get_features"]["features_to_use"]]
        train_y = train[kwargs["get_target"]["target_feature"]]
        model = DecisionTreeClassifier(**hyperparameters)
        model.fit(train_x, train_y)
        logger.info("Final Model has been trained.")
    except KeyError as key_err:
        logger.error("KeyError has occured while training final model: %s", key_err)
        sys.exit(1)
    except Exception as other:
        logger.error("Exception has occured while training final model: %s", other)
        sys.exit(1)
    return model


def train_model_rf(
    hyperparameters: Dict[str, int], train: pd.DataFrame, kwargs: Dict[str, List[str]]
) -> RandomForestClassifier:
    """
    Summary: Trains RandomForest model object
    Function will stop executing if model training fails
    since it is needed further in pipeline.
    Args:
        hyperparameter: Hyperparameters for model training
    """
    try:
        # Finding X and y in train data
        train_x = train[kwargs["get_features"]["features_to_use"]]
        train_y = train[kwargs["get_target"]["target_feature"]]
        model = RandomForestClassifier(**hyperparameters)
        model.fit(train_x, train_y)
        logger.info("Final Model has been trained.")
    except KeyError as key_err:
        logger.error("KeyError has occured while training final model: %s", key_err)
        sys.exit(1)
    except Exception as other:
        logger.error("Exception has occured while training final model: %s", other)
        sys.exit(1)
    return model


def save_model(
    model: Union[RandomForestClassifier, DecisionTreeClassifier], save_path: Path
) -> None:
    """
    Summary: Saves trained model object
    Function will stop executing if saving model fails
    since it is needed further in pipeline.
    Args:
        model: model to be stored as pkl
        save_path: Local path to write model to
    """
    try:
        pickle.dump(model, save_path.open("wb"))
        logger.info("Final Model written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error(
            "While saving model, \
                    File Not Found Error has occured: %s",
            fnfe,
        )
        sys.exit(1)
    except IOError as io_err:
        logger.error(
            "While saving model, \
                    IO Error has occured: %s",
            io_err,
        )
        sys.exit(1)
    except Exception as other:
        logger.error(
            "While saving model, \
                    Other Error has occured: %s",
            other,
        )
        sys.exit(1)
