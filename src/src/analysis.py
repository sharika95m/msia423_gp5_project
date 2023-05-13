import warnings
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def save_figures(features: pd.DataFrame, dir: Path) -> None:
    """Summary: To perform basic exploratory data analysis
    Args:
        features - DataFrame with features to be analyzed
        dir - Path object to save plots
    Returns: No return. All files are saved to disk.
    """
    target = features["class"]

    try:
        figs = []
        count = 0
        for feat in features.columns:
            if feat == "class":
                continue
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.hist([
                features[target == 0][feat].values, \
                        features[target == 1][feat].values
            ], bins = 10)
            ax.set_xlabel(" ".join(feat.split("_")).capitalize())
            ax.set_ylabel("Number of observations")
            figs.append(fig)
            file_name = "image" + str(count) + ".png"
            plt.savefig(dir / file_name)
            count += 1
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
