import logging
from pathlib import Path
import warnings
from typing import Union, Dict
import pandas as pd
import sklearn.metrics
import yaml

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def log_performance(metric: float, comparison: float, metric_name: str) -> None:
    """Summary: Created log messages based on metric value and  provided thresholds
    
    Args:
    metric: Current metric value
    comparison: Threshold value
    metric_name: Name of metric
    """
    if not comparison:
        logger.info('%s = %0.2f. Default threshold not provided ',metric_name,metric)
    elif metric < comparison:
        logger.warning('%s = %0.2f. Metric value less than provided threshold of %0.2f',metric_name,metric,comparison)
    else:
        logger.info('%s = %0.2f. Metric value greater than provided threshold of %0.2f',metric_name,metric,comparison)

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
    Summary: Creates Classification report
    Args:
        scores: dataframe with model predictions and corresponding
        test data labels
        kwargs: dictionary to pass metrics to be tested
    """
    return sklearn.metrics.classification_report(scores[kwargs["truth"]],
            scores[kwargs["label"]])

def evaluate_performance(scores: pd.DataFrame, kwargs: dict)\
                                -> Dict[str, Union[str, float]]:
    """
    Summary: Checks model performance on test dataset using metrics
    Args:
        test_data: dataframe to store test data
        rf_model: Model object used for scoring
        kwargs: dictionary to pass metrics to be tested
    """

    metrics = {}
    #print('Here')
    try:
        # 1. Finding AUC
        curr_auc =  calculate_auc(scores, kwargs["score_type"])
        metrics[kwargs["metric_type"]["auc"]] = curr_auc

        # 2. Finding accuracy
        curr_accuracy = calculate_accuracy(scores,kwargs["score_type"])
        metrics[kwargs["metric_type"]["accuracy"]] = curr_accuracy

        # 3. Finding confusion matrix
        metrics[kwargs["metric_type"]["confusion"]] = \
                        calculate_confusion(scores, kwargs["score_type"])

        # 4. Finding classification report
        metrics[kwargs["metric_type"]["classification_report"]] = \
                        calculate_classification(scores, kwargs["score_type"])
        
        # 5. Testing the thresholds
        threshold_metrics = kwargs.get('threshold',None)

        if not threshold_metrics:
            logger.warning('Threshold values not provided for testing')
        else:
            log_performance(curr_auc,threshold_metrics.get('auc',None),'ROC_AUC_Score')
            log_performance(curr_accuracy,threshold_metrics.get('accuracy',None),'Accuracy')

        logger.info("Metrics have successfully been calculated.")
    except KeyError as k_e:
        logger.error("Metrics are not computed due to KeyError \
                    exception: %s", k_e)
        raise KeyError from k_e
    except ValueError as v_e:
        logger.error("Metrics are not computed due to ValueError \
                    exception: %s", v_e)
        raise ValueError from v_e
    except IndexError as i_e:
        logger.error("Metrics are not computed due to IndexError \
                    exception: %s", i_e)
        raise IndexError from i_e
    except Exception as other:
        logger.error("Metrics are not computed due to an exception: \
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
        logger.info("Metrics written to %s", save_path)
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
                      YAMLError: %s", y_e)
        raise yaml.YAMLError from y_e
    except Exception as other:
        logger.error("Metrics could not be written due to \
                    an exception: %s", other)
        raise Exception from other


# import argparse
# import datetime
# import logging.config
# import yaml
# import pickle
# import create_dataset as cd
# import generate_features as gf
# import train_test_data as ttd
# import crossvalidation as cv
# import model_predict as mp

# logging.config.fileConfig("config/logging/local.conf")
# logger = logging.getLogger("clouds")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Acquire, clean, and create features from clouds data"
#     )
#     parser.add_argument(
#         "--config", default="config/default-config.yaml", help="Path to configuration file"
#     )
#     args = parser.parse_args()

#     # Load configuration file for parameters and run config
#     with open(args.config, "r") as f:
#         try:
#             config = yaml.load(f, Loader=yaml.FullLoader)
#         except yaml.error.YAMLError as e:
#             logger.error("Error while loading configuration from %s", args.config)
#             raise yaml.error.YAMLError from e
#         else:
#             logger.info("Configuration file loaded from %s", args.config)
    
#     run_config = config.get("run_config", {})

#     # Set up output directory for saving artifacts
#     now = int(datetime.datetime.now().timestamp())
#     artifacts = Path(run_config.get("output", "runs")) / str(now)
#     artifacts.mkdir(parents=True)

#     run_config = config.get("run_config", {})

#     dataset_path = Path("data/Telecom Churn Rate Dataset.xlsx")

#     df = cd.get_dataset(dataset_path)
    
#     df_modified, ohe = gf.generate_features(df, config["generate_features"])
    
#     gf.save_dataset(df_modified, artifacts / "modified_data.csv")
#     gf.save_ohe(ohe, artifacts / "ohe_obj.pkl")

#     train, test = ttd.train_test_data_divide(df_modified, config["train_test_data"])

#     train_upsampled = ttd.upsample_train(train, config["train_test_data"])

#     ttd.save_data(train, test, train_upsampled, artifacts)

#     folds = cv.define_folds(config["crossvalidation"]["define_folds"])
#     dt_params, rf_params = cv.get_hyperparameters(config["crossvalidation"]["model_hyperparameters"]) #, scoring_metrics

#     results_dt = cv.gridsearchcv_dt(dt_params, config["crossvalidation"], folds, train_upsampled)
#     results_rf = cv.gridsearchcv_rf(rf_params, config["crossvalidation"], folds, train_upsampled)

#     # model = None
#     if results_dt.best_score_ > results_rf.best_score_:
#         model = cv.train_model_dt(results_dt.best_params_, train_upsampled, config["crossvalidation"])
#         cv.save_model(model, artifacts / "final_model.pkl")
#     else:
#         model = cv.train_model_rf(results_rf.best_params_, train_upsampled, config["crossvalidation"])
#         cv.save_model(model, artifacts / "final_model.pkl")
    
#     model = pickle.load(open(artifacts / "final_model.pkl", "rb"))
#     predicted_data, predictions = mp.model_predict(df_modified, model, config["model_predict"])
#     mp.save_predictions(predicted_data, predictions, artifacts)

#     metrics_perf = evaluate_performance(predictions, config["evaluate_performance"])
#     print(metrics_perf)
#     save_metrics(metrics_perf, artifacts / "metrics.yaml")
    
#     print("Success")