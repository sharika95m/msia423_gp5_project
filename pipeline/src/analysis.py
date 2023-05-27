"""Analysis.py module is too perform basic EDA"""
import warnings
import logging
from pathlib import Path
from typing import Dict
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def eda(data: pd.DataFrame, dir: Path, config: Dict[str, str]) -> None:
    """
    Summary: To perform basic exploratory data analysis
    1. Finding unique customers
    2. Class imbalance (overall and across categories)
    Args:
        features - DataFrame with features to be analyzed
        dir - Path object to save plots
    Returns: No return. All files are saved to disk.
    """
    try:
        file = open(dir / 'basic_eda.txt', 'w')
        n_unique = len(data[config["unique_customers"]].unique())
        file.write(f"The number of unique customers in database: {n_unique}")
        file.write("\n")
        logger.info('Identified number of unique customers')

        file.write("Checking whether the data is imbalanced")
        file.write(data[config["imbalance_check"]].value_counts().\
                reset_index(drop=False).to_string(header=True, index=True))
        file.write("\n")
        logger.info("Have identified that there is class imbalance and \
                    it will need to be balanced during training.")

        for each_col in config["imbalance_check_categories"]:
            check_df = pd.crosstab(data[each_col], \
                data['Churn'])
            check_df['Ratio'] = check_df['Yes']/(check_df['No'] + \
                                check_df['Yes'])
            file.write(check_df.to_string(header=True, index=True))
            file.write("\n")
            logger.info("Have identified that there is class \
                        imbalance across %s.", each_col)
    except FileNotFoundError as fnfe:
        logger.error("While doing basic eda, \
                FileNotFoundError has occured: %s", fnfe)
        raise FileNotFoundError from fnfe
    except KeyError as k_err:
        logger.error("While doing basic EDA, \
                KeyError has occured: %s", k_err)
        raise KeyError from k_err
    except Exception as other:
        logger.error("While doing basic EDA, \
                exception has occured: %s", other)
        raise Exception from other

def save_figures(dataset: pd.DataFrame, dir: Path, config: Dict[str, str]) -> None:
    """
    Summary: To perform basic exploratory data analysis
    using plots
    Args:
        features - DataFrame with features to be analyzed
        dir - Path object to save plots
    Returns: No return. All files are saved to disk.
    """
    dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)
    plot_names = config["plot_names"]
    try:
        for plot_name in plot_names:
            plot_sns = sns.boxplot(data=dataset, \
                x=config[plot_name]["x_feature"], \
                y=config[plot_name]["y_feature"])
            fig = plot_sns.get_figure()
            fig.savefig(dir / plot_name)
        logger.info("All Plots are saved in directory: %s", dir)
    except FileNotFoundError as f_err:
        logger.error("File not found error: %s", f_err)
        raise FileNotFoundError from f_err
    except IOError as i_err:
        logger.error("IO error: %s", i_err)
        raise IOError from i_err
    except Exception as other:
        logger.error("Other error: %s", other)
        raise Exception from other
