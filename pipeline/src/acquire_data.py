import kaggle
from pathlib import Path
import sys
import logging
import warnings
# import time
# import requests

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def get_data(url: str, attempts: int = 4, wait: int = 3, wait_multiple: int = 2) -> str:
    """
    Summary: Acquires data from URL
    Extra args:
    attempts: number of attempts to fetch data
    wait: waiting time before retrying
    wait_multiple: multiplier for wait time between attempts
    ...
    """
    response_text = ""
    #kaggle.api.authenticate()
    kaggle.api.dataset_download_files('datasets/datazng/telecom-company-churn-rate-call-center-data', path=".", unzip = True)
    # for attempt in range(attempts):
    #     try:
    #         res = od.download(url)
    #         if res.status_code == 200:
    #             response_text = res.text
    #             logger.info("URL has been successfully accessed")
    #             break
    #     except requests.exceptions.HTTPError as http_err:
    #         logger.error("HTTP error occurred while trying to get \
    #                     data from url: %s", http_err)
    #         sys.exit(1)
    #     except requests.exceptions.JSONDecodeError as json_decode_err:
    #         logger.error("JSON decode error occurred while trying to \
    #                     get data from url: %s", json_decode_err)
    #         sys.exit(1)
    #     except requests.exceptions.InvalidJSONError as invalid_json_err:
    #         logger.error("Invalid JSON error occurred while trying \
    #                     to get data from url: %s", invalid_json_err)
    #         sys.exit(1)
    #     except requests.exceptions.RequestException as req_err:
    #         logger.error("HTTP error occurred while trying to get data \
    #                     from url: %s", http_err)
    #         if attempt< attempts - 1:
    #             wait_time = wait * (wait_multiple ** attempt)
    #             time.sleep(wait_time)
    #             continue
    #         logger.error("Failed to get data from URL because of error: \
    #                     %s", req_err)
    #         sys.exit(1)
    #     except Exception as err:
    #         logger.error("Other error occurred while trying to get data from \
    #                     url: %s", err)
            # sys.exit(1)
    return 'Success'

def write_data(contents: str, save_path: Path) -> None:
    """
    Summary: Writes data to specified file
    Args:
        url_contents: Text obtained from URL Call
        save_path: Local path to write data to
    """
    try:
        with open(save_path, "w") as f:
            f.write(contents)
            logger.info("Data written to %s", save_path)
    except FileNotFoundError as fnfe:
        logger.error("While writing data, FileNotFoundError \
                    has occured: %s", fnfe)
        sys.exit(1)
    except IOError as i_err:
        logger.error("While writing data, IO error has \
                    occured: %s", i_err)
        sys.exit(1)
    except Exception as e:
        logger.error("Error occurred while trying to \
                    write dataset to file: %s", e)
        sys.exit(1)

def acquire_data(save_path: Path) -> None:
    """
    url: str, 
    Summary: Acquires data from specified URL
    Args:
        url: URL for where data to be acquired is stored
        save_path: Local path to write data to
    """
    url = "kaggle datasets download -d datazng/telecom-company-churn-rate-call-center-data"
    url_contents = get_data(url)
    write_data(url_contents, save_path)

if __name__ == '__main__':
    print(get_data('blah.com'))
