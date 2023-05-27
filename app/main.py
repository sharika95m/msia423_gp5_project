import boto3
import io
import pandas as pd
import json
import sys
import streamlit as st
import time
import argparse
import logging.config
import yaml
from pathlib import Path
from os import path



log_conf_path = path.join(path.dirname(path.abspath(__file__)), 'config/logging/local.conf')
config_path = path.join(path.dirname(path.abspath(__file__)), 'config/app-config.yaml')
logging.config.fileConfig(log_conf_path, disable_existing_loggers=True)
logger = logging.getLogger("app")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config", default=config_path, help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
            raise yaml.error.YAMLError from e
        else:
            logger.info("Configuration file loaded from %s", args.config)

    aws_config = config['aws']


    bucket_name = aws_config.get('bucket_name','check')
    test_file = aws_config.get('output_file_name','check')
    test_location = aws_config.get('output_folder_name','check') + '/' + test_file


    container_1 = st.container()
    col1,col2 = container_1.columns(2)
    col3,col4 = st.columns(2)

    with col1:
        cust_id = st.text_input("Enter Customer ID") ## textbox

    ## Place holders
    col3.write("Customer ID") 
    col3.write("Gender")
    col3.write("Internet Service")
    col3.write("Contract")
    col3.write("Payment Method")

    @st.cache_data(show_spinner=False)
    def get_data_from_s3(bucket_name,test_file):
        session = boto3.Session()
        s3 = session.client("s3")
        obj = s3.get_object(Bucket=bucket_name, Key=test_file)
        data = obj['Body'].read()
        df_back = pd.read_csv(io.BytesIO(data))
        return df_back

    if cust_id:

        ## Accessing the file in S3
        try:
            with st.spinner('Extracting the data from database'):
                df_back = get_data_from_s3(bucket_name,test_location)
        except Exception as e:
            logger.error('Could not download artifacts from aws. Aborting')
            st.error('Error in loading data: Aborting', icon="🚨")
            sys.exit(1)
        else:
            logger.info('File %s obtained from s3 bucket %s',test_location,bucket_name)

        ## FIltering the values
        df_select = df_back[df_back["customerID"]==cust_id]

        ## Checking if customer ID is missing
        if(df_select.shape[0]==0):
            logger.error('Customer ID %s is missing.',cust_id)
            st.error('Customer ID is missing in database')
            cust_id = None
        else:
            check_pred = df_select['pred_probability'].values[0]
            check_pred = (check_pred) * 100
            check_pred = f'{check_pred:.2f}' + '%'

            col2.metric(label="Churn Probability", value= check_pred)
            col4.write(cust_id)
            col4.write(df_select['gender'].values[0])
            col4.write(df_select['InternetService'].values[0])
            col4.write(df_select['Contract'].values[0])
            col4.write(df_select['PaymentMethod'].values[0])
            logger.info('Deatils of Customer ID %s is retrieved.',cust_id)

