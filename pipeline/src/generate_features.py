import sys
import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Union, Dict
from pathlib import Path
import pickle

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def remove_na(df: pd.DataFrame, kwargs: Dict[str, str]) -> pd.DataFrame:
    """
    Summary: Function to remove NAs
    Function will stop execution if there is an error
    since this is an important variable in modelling.
    Args:
        df: Pandas Dataframe with exisiting features
        kwargs: Dictionary storing feature names for computation
    """
    try:
        df_2 = df.copy()
        #Replace empty strings with NA
        df_2[kwargs["column_name"]] = df_2[kwargs["column_name"]].replace(' ',np.nan)

        #Drop all NA values
        df_2 = df_2.dropna(how='any').reset_index(drop=True)
    except KeyError as key_err:
        logger.error("While removing NA, \
                    Key error occurred: %s", key_err)
        sys.exit(1)
    except TypeError as t_err:
        logger.error("While removing NA, \
                    Type error has occured: %s", t_err)
        sys.exit(1)
    except ValueError as v_err:
        logger.error("While removing NA, \
                    Value error has occured: %s", v_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While removing NA, \
                    Other error occurred: %s", other)
        sys.exit(1)

    return df_2

def get_ohe(df: pd.DataFrame, kwargs: Dict[str, str]) -> Union[pd.DataFrame, OneHotEncoder]:
    """
    Summary: Function to One Hot Encode all features 
    and return updated dataframe and OneHotEncoder object
    Args:
        df: Pandas Dataframe with exisiting features
        kwargs: Dictionary storing feature names for one hot encoding
    """
    try:
        ohe = OneHotEncoder(sparse=False,categories="auto",drop="first")
        ohe.fit(df[kwargs["column_names"]])
        temp_df = pd.DataFrame(data=ohe.transform(df[kwargs["column_names"]]), columns=ohe.get_feature_names_out())
        df = pd.concat([df.reset_index(drop=True), temp_df], axis=kwargs["axis"])
    except KeyError as key_err:
        logger.error("While onehotencoding, \
                    Key error occurred: %s", key_err)
        raise KeyError from key_err
    except TypeError as t_err:
        logger.error("While onehotencoding, \
                    Type error has occured: %s", t_err)
        raise TypeError from t_err
    except ValueError as v_err:
        logger.error("While onehotencoding, \
                    Value error has occured: %s", v_err)
        raise ValueError from v_err
    except Exception as other:
        logger.error("While onehotencoding, \
                    Other error occurred: %s", other)
        raise Exception from other

    return df, ohe

def drop_cols(df: pd.DataFrame, kwargs: Dict[str, str])\
                            -> pd.DataFrame:
    """
    Summary: Function to calculate log transform
    Function will stop execution if there is an error
    since this is an important variable in modelling.
    Args:
        df: Pandas Dataframe with exisiting features
        kwargs: Dictionary storing feature names for computation
    """
    try:
        df_2 = df.drop(columns = kwargs["column_name"], axis = int(kwargs["axis"]))
    except KeyError as key_err:
        logger.error("While dropping unnecessary columns, \
                    Key error occurred: %s", key_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While dropping unnecessary columns, \
                    Other error occurred: %s", other)
        sys.exit(1)

    return df_2

def generate_features(data: pd.DataFrame, kwargs: Dict[str, str]) -> pd.DataFrame:
    """
    Summary: Feature Engineering: Create new features
    from exisiting features
    Args:
        data: Pandas Dataframe with exisiting features
        kwargs: New feature details
    """
    # Create a copy of data as features
    features = data.copy()

    # Generating additional features, transformations, and interactions
    # 1. Remove all NAs
    features = remove_na(features, kwargs["remove_na"])

    # 2. One Hot Encode all categorical features
    features, ohe = get_ohe(features,
                            kwargs["get_ohe"])

    # # 3. Drop unnecessary columns
    # features = drop_cols(features,
    #                 kwargs["drop_cols"])

    logger.info("All features have been created successfully.")

    return features, ohe

def save_dataset(data: pd.DataFrame, save_path: Path) -> None:
    """
    Summary: Save modified dataset to directory mentioned in save_path
    Arg:
        data: Modified dataset after cleaning, dropping unnecessary columns,
            and one-hot-encoding
        save_path: directory where files need to be saved.
    """
    try:
        data.to_csv(save_path, index=False)
        logger.info("Modified Dataset written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("While writing modified dataset, FileNotFoundError \
                        has occured: %s", fnfe)
        sys.exit(1)
    except IOError as io_err:
        logger.error("While writing modified dataset, IO Error \
                    has occured: %s", io_err)
        sys.exit(1)
    except Exception as e:
        logger.error("While writing modified dataset, Other \
                    Error has occurred: %s", e)
        sys.exit(1)
    
def save_ohe(ohe: OneHotEncoder, save_path: Path) -> None:
    """
    Summary: Save one hote encoder object to directory
    mentioned in save_path
    Arg:
        data: One hote encoder object
        save_path: directory where files need to be saved.
    """
    try:
        pickle.dump(ohe, open(save_path, 'wb'))
        logger.info("OneHotEncoder is written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("While writing OneHotEncoder, FileNotFoundError \
                        has occured: %s", fnfe)
        sys.exit(1)
    except IOError as io_err:
        logger.error("While writing OneHotEncoder, IO Error \
                    has occured: %s", io_err)
        sys.exit(1)
    except Exception as e:
        logger.error("While writing OneHotEncoder, Other Error \
                    has occurred: %s", e)
        sys.exit(1)
