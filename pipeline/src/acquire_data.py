"""
acquire_data.py

This module provides functions for writing data to a specified file.

Author: Team5
"""

from pathlib import Path
import sys
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def write_data(contents: str, save_path: Path) -> None:
    """
    Summary: Writes data to specified file
    Args:
        url_contents: Text obtained from URL Call
        save_path: Local path to write data to
    """
    try:
        with open(save_path, "w", encoding="utf-8") as open_file:
            open_file.write(contents)
            logger.info("Data written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("While writing data, FileNotFoundError \
                    has occured: %s", fnfe)
        sys.exit(1)
    except IOError as i_err:
        logger.error("While writing data, IO error has \
                    occured: %s", i_err)
        sys.exit(1)
    except Exception as exception_error:
        logger.error("Error occurred while trying to \
                    write dataset to file: %s", exception_error)
        sys.exit(1)
