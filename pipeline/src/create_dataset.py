"""
create_dataset.py

This module provides functions for creating and saving datasets from files.

Author: Team5
"""

import logging
from pathlib import Path
import sys
import warnings
import xlrd
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def get_dataset(file_path: Path) -> pd.DataFrame:
    """
    Summary: Creates dataset from file and returns pandas dataframe
    Args:
        file: Contains the information retrieved from URL
    """
    try:
        df_raw = pd.read_excel(file_path)
    except xlrd.biffh.XLRDError as xlrd_err:
        logger.error("While opening dataset, it raised XLRDError: %s", xlrd_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While opening dataset, other error has occured: %s", other)
        sys.exit(1)
    return df_raw

def save_dataset(df_dataframe: pd.DataFrame, save_path: Path) -> None:
    """Saves dataset to CSV file
    Args:
        url: Pandas Dataframe to be converted to csv
        save_path: Local path to write csv to
    """
    try:
        df_dataframe.to_csv(save_path, index = False)
        logger.info("Dataset written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("File Not Found Error has occured: %s", fnfe)
        sys.exit(1)
    except IOError as io_err:
        logger.error("IO Error has occured: %s", io_err)
        sys.exit(1)
    except Exception as exception_error:
        logger.error("Other Error has occurred: %s", exception_error)
        sys.exit(1)
