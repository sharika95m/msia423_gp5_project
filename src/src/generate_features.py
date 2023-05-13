import sys
import logging
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def calculate_norm_range(df: pd.DataFrame, kwargs: dict[str, str]) -> pd.Series:
    """
    Summary: Function to calculate normalized range
    Function will stop execution if there is an error
    since this is an important variable in modelling.
    Args:
        df: Pandas Dataframe with exisiting features
        kwargs: Dictionary storing feature names for computation
    """
    try:
        return_series = (df[kwargs["max_col"]] -
                        df[kwargs["min_col"]])/df[kwargs["mean_col"]]
    except KeyError as key_err:
        logger.error("While calculating norm range, \
                    Key error occurred: %s", key_err)
        sys.exit(1)
    except TypeError as t_err:
        logger.error("While calculating norm range, \
                    Type error has occured: %s", t_err)
        sys.exit(1)
    except ValueError as v_err:
        logger.error("While calculating norm range, \
                    Value error has occured: %s", v_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While calculating norm range, \
                    Other error occurred: %s", other)
        sys.exit(1)

    return return_series

def calculate_range(df: pd.DataFrame, kwargs: dict[str, str]) -> pd.Series:
    """
    Summary: Function to calculate range
    Args:
        df: Pandas Dataframe with exisiting features
        kwargs: Dictionary storing feature names for computation
    """
    try:
        return_series = df[kwargs["max_col"]] - df[kwargs["min_col"]]
    except KeyError as key_err:
        logger.error("While calculating IR range, \
                    Key error occurred: %s", key_err)
        raise KeyError from key_err
    except TypeError as t_err:
        logger.error("While calculating IR range, \
                    Type error has occured: %s", t_err)
        raise TypeError from t_err
    except ValueError as v_err:
        logger.error("While calculating IR range, \
                    Value error has occured: %s", v_err)
        raise ValueError from v_err
    except Exception as other:
        logger.error("While calculating IR range, \
                    Other error occurred: %s", other)
        raise Exception from other

    return return_series

def calculate_log_transform(df: pd.DataFrame, kwargs: dict[str, str])\
                            -> pd.Series:
    """
    Summary: Function to calculate log transform
    Function will stop execution if there is an error
    since this is an important variable in modelling.
    Args:
        df: Pandas Dataframe with exisiting features
        kwargs: Dictionary storing feature names for computation
    """
    try:
        return_series = df[kwargs["log_entropy"]].apply(np.log)
    except KeyError as key_err:
        logger.error("While calculating log transform, \
                    Key error occurred: %s", key_err)
        sys.exit(1)
    except TypeError as t_err:
        logger.error("While calculating log transform, \
                    Type error has occured: %s", t_err)
        sys.exit(1)
    except AttributeError as a_err:
        logger.error("While calculating log transform, \
                    Attribute error has occured: %s", a_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While calculating log transform, \
                    Other error occurred: %s", other)
        sys.exit(1)

    return return_series

def calculate_entropy(df: pd.DataFrame, kwargs: dict[str, str]) -> pd.Series:
    """
    Summary: Function to calculate entropy
    Function will stop execution if there is an error
    since this is an important variable in modelling.
    Args:
        df: Pandas Dataframe with exisiting features
        kwargs: Dictionary storing feature names for computation
    """
    try:
        return_series = df[kwargs["col_a"]].multiply(df[kwargs["col_b"]])
    except KeyError as key_err:
        logger.error("While calculating entropy, \
                    Key error occurred: %s", key_err)
        sys.exit(1)
    except TypeError as t_err:
        logger.error("While calculating entropy, \
                    Type error has occured: %s", t_err)
        sys.exit(1)
    except AttributeError as a_err:
        logger.error("While calculating entropy, \
                    Attribute error has occured: %s", a_err)
        sys.exit(1)
    except ValueError as v_err:
        logger.error("While calculating entropy, \
                    Value error has occured: %s", v_err)
        sys.exit(1)
    except Exception as other:
        logger.error("While calculating entropy, \
                    Other error occurred: %s", other)
        sys.exit(1)

    return return_series

def generate_features(data: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
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
    # 1. Calculating IR_norm_range
    features["IR_norm_range"] = calculate_norm_range(features,
                    kwargs["calculate_norm_range"]["IR_norm_range"])

    # 2. Calculating IR_range
    features["IR_range"] = calculate_range(features,
                    kwargs["calculate_range"]["IR_range"])

    # 3. Calculating log_entropy
    features["log_entropy"] = calculate_log_transform(features,
                    kwargs["calculate_log_transform"]["log_transform"])

    # 4. Calculating entropy_x_contrast
    features["entropy_x_contrast"] = calculate_entropy(features,
                    kwargs["calculate_entropy"]["entropy_x_contrast"])

    logger.info("All features have been created successfully.")

    return features
