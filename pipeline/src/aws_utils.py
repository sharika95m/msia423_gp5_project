from pathlib import Path
import sys
import os
import warnings
import logging
import boto3
import botocore

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def get_s3_file_path(config: dict) -> Path:
    """
    Summary: Returns a Pathlib Path object to the file in the specified S3 bucket and folder.
    
    Args:
        config: A dictionary with three keys:
                 - "bucket_name": Name of the AWS S3 bucket.
                 - "dataset_folder_name": Name of the folder within the S3 bucket.
                 - "dataset_file_name": Name of the xlsx file within the folder.
        
    Returns:
        A Pathlib Path object to the file in the S3 bucket.
    """
    try:
        s3 = boto3.resource('s3')
        s3.meta.client.head_bucket(Bucket=config["bucket_name"])

        # Check if the bucket exists
        bucket = s3.Bucket(config["bucket_name"])
        if not bucket.creation_date:
            logger.error(f"Bucket {config['bucket_name']} does not exist")
            raise ValueError(f"Bucket {config['bucket_name']} does not exist")

        # Check if the folder exists
        folder_obj = list(bucket.objects.filter(Prefix=f"{config['dataset_folder_name']}/"))
        if not folder_obj:
            logger.error(f"Folder {config['dataset_folder_name']} does not exist in bucket {config['bucket_name']}")
            raise ValueError(f"Folder {config['dataset_folder_name']} does not exist in bucket {config['bucket_name']}")

        # Check if the file exists
        file_obj = s3.Object(config["bucket_name"], f"{config['dataset_folder_name']}/{config['dataset_file_name']}")
        try:
            file_obj.load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.error(f"File {config['dataset_file_name']} does not exist in bucket {config['bucket_name']}/{config['dataset_folder_name']}")
                raise ValueError(f"File {config['dataset_file_name']} does not exist in bucket {config['bucket_name']}/{config['dataset_folder_name']}")
            else:
                raise

        path_to_file = Path(f"s3://{config['bucket_name']}/{config['dataset_folder_name']}/{config['dataset_file_name']}")
        logger.info(f"Successfully created Path object to file {config['dataset_file_name']} in bucket {config['bucket_name']}")
        return path_to_file

    except botocore.exceptions.BotoCoreError as err:
        logger.error(f"Error occurred while creating path to file {config['dataset_file_name']}: {err}")
        raise err from None
    except botocore.exceptions.NoCredentialsError as err:
        logger.error(f"No AWS credentials found: {err}")
        raise err from None
    except Exception as err:
        logger.error(f"Unexpected error occurred: {err}")
        raise err from None

def upload_artifacts(artifacts: Path, config: dict) -> list[str]:
    """
    Summary: Upload all the artifacts in the specified directory to S3
    Args:
        artifacts: Directory containing all the artifacts from a given
        experiment
        config: Config required to upload artifacts to S3
    Returns:
        List of S3 uri's for each file that was uploaded
    """
    try:
        s3_session = boto3.Session()
        s3_conn = s3_session.client("s3")

        dir_files = os.listdir(artifacts)

        exisiting_buckets = [bucket.name for bucket in boto3.resource('s3').buckets.all()]

        if config["bucket_name"] in exisiting_buckets:
            for file_name in dir_files:
                file_s3_name = (artifacts / file_name).as_posix()
                if "figures" not in file_s3_name:
                    s3_conn.upload_file(file_s3_name,
                                config["bucket_name"], file_s3_name)
                else:
                    figure_names = os.listdir(artifacts/file_name)
                    for figure in figure_names:
                        figure_s3_name = (artifacts/file_name/figure).as_posix()
                        s3_conn.upload_file(figure_s3_name,
                                config["bucket_name"], figure_s3_name)
            logger.info("Artifacts have successfully been \
                        uploaded to S3 bucket.")
        else:
            logger.error("Bucket is not created. Please create bucket \
                        with name %s on your AWS Console",
                        config["bucket_name"])
            sys.exit(SystemExit)
    except FileNotFoundError as f_err:
        logger.error("File Not Found Error has occured: %s", f_err)
        raise FileNotFoundError from f_err
    except botocore.exceptions.NoCredentialsError as cred_err:
        logger.error("No Credentials error has occured: %s", cred_err)
        raise botocore.exceptions.NoCredentialsError from cred_err
    except botocore.exceptions.ClientError as client_err:
        logger.error("Client Error has occured: %s", client_err)
        raise botocore.exceptions.ClientError from client_err
    except botocore.exceptions.ParamValidationError as p_err:
        logger.error("Parameter Validation Error has occured: %s", p_err)
        raise botocore.exceptions.ParamValidationError from p_err
    except Exception as other:
        logger.error("Other Error has occured: %s", other)
        raise Exception from other
