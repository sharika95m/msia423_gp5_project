import argparse
import datetime
import logging.config
from pathlib import Path
import pickle

import yaml

import src.create_dataset as cd
import src.generate_features as gf
import src.train_test_data as ttd
import src.crossvalidation as cv
import src.model_predict as mp
import src.evaluate_performance as ep
import src.aws_utils as aws
from os import path

log_conf_path = path.join(path.dirname(path.abspath(__file__)), 'config/logging/local.conf')
config_path = path.join(path.dirname(path.abspath(__file__)), 'config/default-config.yaml')
logging.config.fileConfig(log_conf_path, disable_existing_loggers=True)
logger = logging.getLogger("clouds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config", default=config_path, help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
            raise yaml.error.YAMLError from e
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Acquire data from online repository and save to disk
    # ad.acquire_data(run_config["data_source"], artifacts / "clouds.data")

    #Create dataset from raw data found within s3
    dataset_path2 = aws.get_s3_file_path(config["aws"])
    print(dataset_path2)
    #df = cd.get_dataset(dataset_path)

    # Create structured dataset from raw data; save to disk    
    # dataset_path = Path("data/Telecom Churn Rate Dataset.xlsx")
    df = cd.get_dataset(dataset_path2)

    # Enrich dataset and OneHotEncoder with features for model training; save to disk
    df_modified, ohe = gf.generate_features(df, config["generate_features"])
    gf.save_dataset(df_modified, artifacts / "modified_data.csv")
    gf.save_ohe(ohe, artifacts / "ohe_obj.pkl")

    # # Generate statistics and visualizations for summarizing the data; save to disk
    # figures = artifacts / "figures"
    # figures.mkdir()
    # eda.save_figures(features, figures)

    # Created train-test dataset and then upsample the train; save to disk
    train, test = ttd.train_test_data_divide(df_modified, config["train_test_data"])
    train_upsampled = ttd.upsample_train(train, config["train_test_data"])
    ttd.save_data(train, test, train_upsampled, artifacts)

    # Perform cross-validation
    # 1. Create folds | 2. Get hyperparameters for tuning
    # 3. Find best model | 4. Save best model
    folds = cv.define_folds(config["crossvalidation"]["define_folds"])
    dt_params, rf_params = cv.get_hyperparameters(config["crossvalidation"]["model_hyperparameters"])
    results_dt = cv.gridsearchcv_dt(dt_params, config["crossvalidation"], folds, train_upsampled)
    results_rf = cv.gridsearchcv_rf(rf_params, config["crossvalidation"], folds, train_upsampled)
    if results_dt.best_score_ > results_rf.best_score_:
        model = cv.train_model_dt(results_dt.best_params_, train_upsampled, config["crossvalidation"])
        cv.save_model(model, artifacts / "final_model.pkl")
    else:
        model = cv.train_model_rf(results_rf.best_params_, train_upsampled, config["crossvalidation"])
        cv.save_model(model, artifacts / "final_model.pkl")
    
    # Make predictions with trained model object on entire dataset
    model = pickle.load(open(artifacts / "final_model.pkl", "rb"))
    predicted_data, predictions = mp.model_predict(df_modified, model, config["model_predict"])
    mp.save_predictions(predicted_data, predictions, artifacts)

    # Evaluate model performance metrics; save metrics to disk
    metrics = ep.evaluate_performance(predictions, config["evaluate_performance"])
    ep.save_metrics(metrics, artifacts / "metrics.yaml")

    # Upload all artifacts to S3
    aws_config = config.get("aws")
    if aws_config.get("upload", False):
        aws.upload_artifacts(artifacts, aws_config)
