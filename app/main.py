"""
Driver code to implement the streamlit customer representative 
facing app
"""

import io
import sys
import argparse
import logging.config
from os import path
import base64
import pandas as pd
import boto3
import yaml
import streamlit as st


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

    st.set_page_config(page_title='Retention Wizard', page_icon='🧙‍♂️')

    def add_bg_from_local(image_file: str):
        """
        Summary: Creates a markdown to display the background image
        Args:
        image_file: Location of the image in directory
        """

        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read())
            st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, 0.5), \
                    rgba(255, 255, 255, 0.5)),\
                        url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    add_bg_from_local('resources/background.jpeg')

    st.header('Retention Wizard')
    container_1 = st.container()
    col1,col2 = container_1.columns(2)
    col3,col4 = st.columns(2)

    with col1:
        CUST_ID = st.text_input("Enter Customer ID") ## textbox

    container_1.text('')

    ## Place holders

    @st.cache_data(show_spinner=False)
    def get_data_from_s3(bucket,file):
        """
        Summary: Function to get the data from an AWS Bucket
        Args:
        bucket: Name of the AWS bucket
        file: Name of the file in the AWS bucket
        """
        session = boto3.Session()
        s3_session = session.client("s3")
        obj = s3_session.get_object(Bucket=bucket, Key=file)
        data = obj['Body'].read()
        df_data = pd.read_csv(io.BytesIO(data))
        return df_data

    if CUST_ID:

        ## Accessing the file in S3
        try:
            with st.spinner('Extracting the data from database'):
                df_back = get_data_from_s3(bucket_name,test_location)
        except Exception as e:
            logger.error('Could not download artifacts from aws. Aborting')
            st.error('Error in loading data: Aborting')
            sys.exit(1)
        else:
            logger.info('File %s obtained from s3 bucket %s',test_location,bucket_name)

        ## FIltering the values
        df_select = df_back[df_back["customerID"]==CUST_ID]

        ## Checking if customer ID is missing
        if df_select.shape[0]==0:
            logger.error('Customer ID %s is missing.',CUST_ID)
            st.error('Customer ID is missing in database')
            CUST_ID = None
        else:
            check_pred = df_select['pred_probability'].values[0]
            check_pred = (check_pred) * 100
            check_pred = f'{check_pred:.2f}' + '%'

            col3.write("Customer ID")
            col3.write("Gender")
            col3.write("Phone Service")
            col3.write("Internet Service")
            col3.write("Contract")
            col3.write("Tenure")
            col3.write("Monthly Charges")
            col3.write("Total Charges")

            col2.metric(label="Churn Probability", value= check_pred)
            col4.write(CUST_ID)
            col4.write(df_select['gender'].values[0])
            col4.write(df_select['PhoneService'].values[0])
            col4.write(df_select['InternetService'].values[0])
            col4.write(df_select['Contract'].values[0])
            col4.write(str(df_select['tenure'].values[0]))
            col4.write(str(df_select['MonthlyCharges'].values[0]))
            col4.write(str(df_select['TotalCharges'].values[0]))
            logger.info('Details of Customer ID %s is retrieved.',CUST_ID)
