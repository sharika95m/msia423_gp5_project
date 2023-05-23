import pytest
import os
from pathlib import Path
import boto3
from moto import mock_s3
from src import aws_utils
#from aws_utils import get_s3_file_path, upload_artifacts 

from src.aws_utils import get_s3_file_path
from src.aws_utils import upload_artifacts

@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'

@pytest.fixture
def s3(aws_credentials):
    with mock_s3():
        yield boto3.client('s3', region_name='us-east-1')


def list_s3_objects(bucket_name: str, s3_client) -> list:
    """Utility function to list all objects in an S3 bucket"""
    s3_objects = []
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        s3_objects = [obj['Key'] for obj in response['Contents']]
    return s3_objects



def test_get_s3_file_path(s3):
    config = {
        "bucket_name": "test_bucket",
        "dataset_folder_name": "test_folder",
        "dataset_file_name": "test_file.xlsx",
    }
    
    s3.create_bucket(Bucket=config["bucket_name"])
    s3.put_object(Bucket=config["bucket_name"], Key=f"{config['dataset_folder_name']}/{config['dataset_file_name']}", Body=b'test')
    
    assert get_s3_file_path(config) == Path(f"s3://{config['bucket_name']}/{config['dataset_folder_name']}/{config['dataset_file_name']}")

def test_upload_artifacts(s3, tmp_path):
    config = {
        "bucket_name": "test_bucket",
    }

    s3.create_bucket(Bucket=config["bucket_name"])

    # Create temporary directory with files
    d = tmp_path / "artifacts"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("content")

    # Run the function under test
    upload_artifacts(d, config)
    
    # Get the list of files in the S3 bucket after uploading
    s3_files_after = list_s3_objects(config["bucket_name"], s3)
    
    # Get the filename of the uploaded file
    uploaded_file_path = p.name
    
    # Check if any of the paths in the S3 bucket ends with the uploaded file name
    assert any(s3_file.endswith(uploaded_file_path) for s3_file in s3_files_after)



