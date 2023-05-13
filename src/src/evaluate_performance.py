import logging
from pathlib import Path
import warnings
from typing import Union
import pandas as pd
import sklearn.metrics
import yaml

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def calculate_auc(scores: pd.DataFrame, kwargs: dict) -> float:
    """
    Summary: Calculates AUC
    Args:
        scores: dataframe with model predictions and corresponding
        test data labels
        kwargs: dictionary to pass metrics to be tested
    """
    return float(sklearn.metrics.roc_auc_score(scores[kwargs["truth"]],
                scores[kwargs["probability"]]))

def calculate_accuracy(scores: pd.DataFrame, kwargs: dict) -> float:
    """
    Summary: Calculates accuracy
    Args:
        scores: dataframe with model predictions and corresponding
        test data labels
        kwargs: dictionary to pass metrics to be tested
    """
    return float(sklearn.metrics.accuracy_score(scores[kwargs["truth"]],
                scores[kwargs["label"]]))

def calculate_confusion(scores: pd.DataFrame, kwargs: dict) -> str:
    """
    Summary: Calculates Confusion Matrix
    Args:
        scores: dataframe with model predictions and corresponding
        test data labels
        kwargs: dictionary to pass metrics to be tested
    """
    return pd.DataFrame(
            sklearn.metrics.confusion_matrix(scores[kwargs["truth"]],
            scores[kwargs["label"]]), index=["Actual negative",
            "Actual positive"],columns=["Predicted negative",
            "Predicted positive"]).to_string(header = True, index = True)

def calculate_classification(scores: pd.DataFrame, kwargs: dict) -> str:
    """
    Summary: Calculates Classification report
    Args:
        scores: dataframe with model predictions and corresponding
        test data labels
        kwargs: dictionary to pass metrics to be tested
    """
    return sklearn.metrics.classification_report(scores[kwargs["truth"]],
            scores[kwargs["label"]])

def evaluate_performance(scores: pd.DataFrame, kwargs: dict)\
                                -> dict[str, Union[str, float]]:
    """
    Summary: Checks model performance on test dataset using metrics
    Args:
        test_data: dataframe to store test data
        rf_model: Model object used for scoring
        kwargs: dictionary to pass metrics to be tested
    """

    metrics = {}
    try:
        # 1. Finding AUC
        metrics[kwargs["metric_type"]["auc"]] = \
                        calculate_auc(scores, kwargs["score_type"])

        # 2. Finding accuracy
        metrics[kwargs["metric_type"]["accuracy"]] = \
                        calculate_accuracy(scores,kwargs["score_type"])

        # 3. Finding confusion matrix
        metrics[kwargs["metric_type"]["confusion"]] = \
                        calculate_confusion(scores, kwargs["score_type"])

        # 4. Finding classification report
        metrics[kwargs["metric_type"]["classification_report"]] = \
                        calculate_classification(scores, kwargs["score_type"])

        logger.info("Metrics have successfully been calculated.")
    except KeyError as k_e:
        logger.error("Metrics is not be computed due to KeyError \
                    exception: %s", k_e)
        raise KeyError from k_e
    except ValueError as v_e:
        logger.error("Metrics is not be computed due to ValueError \
                    exception: %s", v_e)
        raise ValueError from v_e
    except IndexError as i_e:
        logger.error("Metrics is not be computed due to IndexError \
                    exception: %s", i_e)
        raise IndexError from i_e
    except Exception as other:
        logger.error("Metrics is not be computed due to an exception: \
                    %s", other)
        raise Exception from other

    return {"metrics": metrics}

def save_metrics(metrics: dict, save_path: Path) -> None:
    """
    Summary: Saves metrics dictionary to YAML file
    Args:
        metrics: dictionary to be stored as yaml
        save_path: Local path to write yaml to
    """
    try:
        with open(save_path,"w") as file:
            yaml.dump(metrics,file)
        logger.info("Dictionary written to %s", save_path)
    except TypeError as t_e:
        logger.error("Metrics could not be written due to \
                    TypeError: %s", t_e)
        raise TypeError from t_e
    except FileNotFoundError as fnfe:
        logger.error("Metrics could not be written due to \
                    FileNotFoundError: %s", fnfe)
        raise FileNotFoundError from fnfe
    except PermissionError as p_e:
        logger.error("Metrics could not be written due to \
                    PermissionError: %s", p_e)
        raise PermissionError from p_e
    except yaml.YAMLError as y_e:
        logger.error("Metrics could not be written due to \
                    yaml.YAMLError: %s", y_e)
        raise yaml.YAMLError from y_e
    except Exception as other:
        logger.error("Metrics could not be written due to \
                    an exception: %s", other)
        raise Exception from other
