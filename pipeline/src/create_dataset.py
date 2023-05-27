import logging
from pathlib import Path
import sys
import warnings
import xlrd
import numpy as np
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
