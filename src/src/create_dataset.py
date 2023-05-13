import logging
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def create_dataset(file_path: Path, column_names: list[str]) -> pd.DataFrame:
    """
    Summary: Creates dataset from file and returns pandas dataframe
    Args:
        file: Contains the information retrieved from URL
    """
    try:
        lines = []
        with open(file_path,"r") as file:
            lines = file.readlines()[53:]
        final_lines = lines[:2053]

        # First type of cloud
        first_cloud_text = final_lines[:1024]
        first_cloud_text = [cloud.replace(r"[\s|\t]+"," ")
                            for cloud in first_cloud_text]
        first_cloud_text = [cloud.replace(r"\n","")
                            for cloud in first_cloud_text]

        first_cloud = []
        for cloud in first_cloud_text:
            cloud_data = cloud.split(" ")
            cloud_data_flt = [float(i) for i in cloud_data if i!=""]
            first_cloud += [cloud_data_flt]

        first_cloud = pd.DataFrame(first_cloud, columns=column_names)
        first_cloud["class"] = np.zeros(len(first_cloud), dtype = np.int8)

        #Second type of cloud
        second_cloud_text = final_lines[1029:]
        second_cloud_text = [cloud.replace(r"[\s|\t]+"," ")
                            for cloud in second_cloud_text]
        second_cloud_text = [cloud.replace(r"\n"," ")
                            for cloud in second_cloud_text]

        second_cloud = []
        for cloud in second_cloud_text:
            cloud_data = str(cloud).split(" ")
            cloud_data_flt = [float(i) for i in cloud_data if i!=""]
            second_cloud += [cloud_data_flt]

        second_cloud = pd.DataFrame(second_cloud, columns=column_names)
        second_cloud["class"] = np.ones(len(second_cloud), dtype = np.int8)

        # Full dataset
        data = pd.concat([first_cloud, second_cloud])

        logger.info("Dataset has been successfully created.")
    except FileNotFoundError as f_err:
        logger.error("File Not Found error occurred: %s", f_err)
        sys.exit(1)
    except KeyError as k_err:
        logger.error("Key error occurred: %s", k_err)
        sys.exit(1)
    except AttributeError as a_err:
        logger.error("Attribute error occurred: %s", a_err)
        sys.exit(1)
    except ValueError as val_err:
        logger.error("Value error occurred: %s", val_err)
        sys.exit(1)
    except IndexError as index_err:
        logger.error("Index error occured: %s", index_err)
        sys.exit(1)
    except Exception as other_err:
        logger.error("Other error occurred: %s", other_err)
        sys.exit(1)

    return data

def save_dataset(df: pd.DataFrame, save_path: Path) -> None:
    """Saves dataset to CSV file
    Args:
        url: Pandas Dataframe to be converted to csv
        save_path: Local path to write csv to
    """
    try:
        df.to_csv(save_path, index = False)
        logger.info("Dataset written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("File Not Found Error has occured: %s", fnfe)
        sys.exit(1)
    except IOError as io_err:
        logger.error("IO Error has occured: %s", io_err)
        sys.exit(1)
    except Exception as e:
        logger.error("Other Error has occurred: %s", e)
        sys.exit(1)
