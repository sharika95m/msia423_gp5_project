"""
Summary: Obtain train test data
"""
import sys
import warnings
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train_test_data_divide(
    data: pd.DataFrame, kwargs: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summary: train_test_data_divide function performs 1 key operations
    1. Create train and test dataset
    3. Return train and test datasets to function call
    Args:
        data: dataframe to perform modeling on
        kwargs: Dictionary with information for train-test split
        and model hyperparameters
    """
    try:
        train, test = train_test_split(data, **kwargs["split_data"])

        logger.info("Train-test datasets have been created.")
    except ValueError as v_err:
        logger.error("Value error occured during train-test split: %s", v_err)
        sys.exit(1)
    except KeyError as k_err:
        logger.error("Key error occured during train-test split: %s", k_err)
        sys.exit(1)
    except TypeError as t_err:
        logger.error("Type error occured during train-test split: %s", t_err)
        sys.exit(1)
    except Exception as other_err:
        logger.error("Rrror occured during train-test split: %s", other_err)
        sys.exit(1)

    return train, test


def upsample_train(train: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
    """
    Summary: Upsample training data
    Function will stop executing if upsampling fails.

    Args:
        train_data: dataframe to be upsampled
        save_path: Local path to write csv to
    """
    try:
        train_minority = train[train["Churn_Yes"] == 1]
        train_other = train[train["Churn_Yes"] == 0]

        if train_minority.shape[0] == 0:
            logger.warning("Minority Class, i.e churn cases not present in data")
            return train

        min_upsampled = resample(
            train_minority,
            random_state=kwargs["upsample_train"]["random_state"],
            n_samples=len(train_other) - len(train_minority),
            replace=True,
        )
        train_upsampled = pd.concat(
            [min_upsampled, train],
            axis=kwargs["upsample_train"]["axis"],
            ignore_index=kwargs["upsample_train"]["ignore_index"],
            sort=kwargs["upsample_train"]["sort"],
        )
        logger.info("The train dataset has been upsampled to maintain class balance.")

    except ValueError as v_err:
        logger.error(
            "Value error occured during Upsampling: %s , Returning original dataset",
            v_err,
        )
        return train
    except KeyError as k_err:
        logger.error(
            "Key error occured during Upsampling: %s, Returning original dataset", k_err
        )
        return train
    except TypeError as t_err:
        logger.error(
            "Type error occured during Upsampling: %s, Returning original dataset",
            t_err,
        )
        return train
    except Exception as other_err:
        logger.error(
            "Rrror occured during Upsampling: %s, Returning original dataset", other_err
        )
        return train

    return train_upsampled


def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_upsample: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Summary: Saves train and test dataset to CSV file
    Function will stop executing if save_data fails
    since test dataset is needed further in pipeline.

    Args:
        train_data, test_data: dataframe to be stored as csv
        save_path: Local path to write csv to
    """
    try:
        train_data.to_csv(save_path / "train.csv", index=False)
        test_data.to_csv(save_path / "test.csv", index=False)
        train_upsample.to_csv(save_path / "train_upsample.csv", index=False)
        logger.info(
            "Train, test, upsampled train Datasets \
                    are written to %s",
            save_path,
        )
    except FileNotFoundError as fnfe:
        logger.error(
            "File Not Found Error has occured, \
                    While saving train/test dataset: %s",
            fnfe,
        )
        sys.exit(1)
    except IOError as io_err:
        logger.error(
            "IO Error has occured, \
                    While saving train/test dataset: %s",
            io_err,
        )
        sys.exit(1)
    except Exception as other_err:
        logger.error(
            "Other Error has occured, \
                    While saving train/test dataset: %s",
            other_err,
        )
        sys.exit(1)
