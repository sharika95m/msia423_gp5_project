# Clouds

## Overview

On a high-level, the aim of the project is to analyze and build a Machine Learning model from data on UCI website ([LINK](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data)). This dataset is created from images. It ultimately has 10 features, which describe a particular type of cloud. The objective of the machine learning model would be to use these 10 features to classify the cloud. The project involves the following steps

1. Acquiring the data from website
2. Cleaning the data
3. Creating new features using existing data
4. Building basic visualizations
5. Creating train and test dataset
6. Building a machine learning model to classify the type of cloud
7. Making predictions on a test set
8. Evaluating the model on certain metrics
9. Uploading all project deliverables to an AWS s3 bucket

Before proceeding further, 
- [ ] Please create S3 bucket on your AWS Console with either bucket name of your choice or smv7369-clouds-hw2

- [ ] If you chose to create an S3 bucket with bucket name of your choice, please update the YAML file. Under AWS section, update the bucket_name.

## Project folder structure

All modules are stored inside the src folder. All artifacts from the various modules will be generated inside a folder corresponding to the run inside the runs directory. There is also a testing module inside the tests folder. In order to simplify things at the user's end, all critical information is written inside a **config.yaml** file stored inside the config directory. This information includes URL to acquire data from, the model hyperparameters, training features, test size, etc.

## Brief over of model pipeline

All modules needed to run the machine learning model are called from the pipeline.py file. While running the modules, some module produces various artifacts needed to run the next step in the pipeline. The various modules include:

### Step 1. acquire_data

Acquiring the data from website ([LINK](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data)) and writing this data to '**clouds.data**' file inside the respective run folder inside the runs directory.

#### Instructions

This is done by using the requests package in python to acquire the data from the website. This is done through the get_data() function. The URL information has been set inside **config.yaml** file. If the connection request is successful, then data is written to the **clouds.data** file, which is executed through the write_data() function. 

Is any action needed from the user: **NO**.

### Step 2. create_dataset: 

Takes the data from the previously created '**clouds.data**' file, identifies the value corresponding to each feature and cleans it. Cleaning the data will involve removing whitespaces, converting to float type, etc.

#### Instructions

This is done by using the pandas library in python to organise the previously acquired data. This is done through the create_dataset() function. The entire text inside the **clouds.data** file is split into a list of strings at each new line. The numeric value corresponsing to a record is extracted, striped of any special characters, converted to floating type and then added to a dataframe. Here, the necessary information includes the names of the columns, which is passed through the configuration file. If this is successful, then data is written to the **clouds.csv** file.

Is any action needed from the user: **NO**.

### Step 3. generate_features: 

The generate_features module is critical because it creates 3 important features needed in the machine learning model. This module involves feature engineering to generate 4 features: '_norm\_range_', '_range_', '_log\_entropy_' and '_entropy\_x\_contrast_'. After generating the features, the dataframe is returned to the function call where it updates the previous file. Needed information in this module include name of the columns, which is passed through the config file.

#### Instructions

This is done by using the pandas library in python to organise the previously acquired data.

Due to the criticality of this module, unit tests are created for this module. This is stored inside the tests directory. In order to run the tests, the user must navigate to the tests directory and run ```pytest testing_generate_features.py``` to test. However, Dockerfile created creates images that run both the test and the model.

Is any action needed from the user: **NO**.

### Step 4. analysis:

Usually for a majority of models such as neural networks, logistic regression, etc, it is important to check the normality of the data. In this module, histograms are developed to check the data distribution.

#### Instructions

This is done using the mathplotlib library. The historgrams generates are saved to a figures directory inside the corresponding run in the runs folder.

Is any action needed from the user: **NO**.

### Step 5. train_model

In this module, first, the dataset with generated features are split into train and test data. The test size and random state are specified using the config file. Next, a Random Forest classifier is built using the training data. The hyperparameters for the model is passed through the config file. Finally, all the datasets and model object are saved to the corresponding run folder.

#### Instructions

Ihe operations are executed with sklearn library. The model currently has good performance. Hence, it is not necessary to change the hyperparameters.

Is any action needed from the user: **NO**.

### Step 6. score_model

In this module, the trained model object is used to make predictions on the test data. It returns a dataframe containing the true class labels, the predicted class labels and the probability of those predictions being true.

#### Instructions

This is done using the sklearn and pandas library. Is any action needed from the user: **NO**.

### Step 7. evaluate_performance

In this module, the previously made predictions are used to evaluate the model using metrics mentioned in the config file. This is then written to a yaml file inside the corresponding run directory.

#### Instructions

This is done using the sklearn and pandas library. Is any action needed from the user: **NO**.

### Step 8. aws_utils

This module is used to upload all the generated model artifacts to an AWS S3 bucket. 

#### Instructions

Is any action needed from the user: **YES**. Read below.

In the current set-up, the code assumes that a s3 bucket is created in the user's aws console with the name _smv7369-clouds-hw2_. Upon execution, if this does not exist, it will stop the execution. In order to circumvent this,

1. Log in to your AWS Management Console with your IAM User Credentials ([LINK](https://aws.amazon.com/console/))
2. Search for S3 bucket in your search bar
3. On S3 page, click on Create bucket button.
4. While creating the bucket, you can give a bucket name of choice or maintain the default name _smv7369-clouds-hw2_.
5. Maintain all other default settings (unless you specifically want to change something)
6. Finally, click Create bucket buttom at the bottom.
7. - [ ] **If you have given the bucket a name of your choice, update the config.yaml inside the config directory under the AWS section. See the highlighted name in the image to understand what to change**:

![AWS bucket Name to be Changed in YAML file](./img1.png)

### Step 9. pipeline

The pipeline.py file runs all the modules needed to executed the project by calling them in sequence. It also extracts the informations from the yaml file and send it to the appropriate module.

### Pipeline dependencies

The following project was executed on a virtual environment on the author's PC as well as on a docker container. So the needed python libraries for running the model, running the unit tests and uploading them to AWS S3 bucket have been identified. 

#### Instructions

This has already been created for the user and stored in the **requirements.txt**. If user wishes to install them inside a virtual environment, they can run the following command on the command line inside the virtual environment.

```
pip install -r requirements.txt
```

Ensure you are in the same directory as the **requirements.txt** file. 

However, if the user will be running the model using a container, the Dockerfile already has the requirements.txt specified. Hence, no action is needed from the user.

### AWS Setup

Since the pipeline includes uploading files to an S3 bucket, the user must have an IAM user created on the AWS Console and have it configured on their local system. This configuration will be passed to the s3 client while making the call to access the buckets.

#### Instructions

Is any action needed from the user: **YES**. Read below.

1. Configure an aws profile with the following command on command line(Give a profile name of your choice)

```
aws configure sso --profile <profile_name>
```

2. After this, you need to login via the command on command line (You may need to refresh login every time you close terminal)

```
aws login sso --profile <profile_name>
```

3. Verify the identity with the command on command line:

```
aws sts get-caller-identity --profile <profile_name>
```

4. Create a credentials file to authenticate the calls to boto3 with the following command on command line

```
export AWS_PROFILE=personal-sso-admin
```

### Running model on a docker container

The docker containers allow an efficient way to run different applications on different computing environments. In order to build a container image from the Dockerfile and run a container using that created image, read the instructions. 

Currently, the Dockerfile will run both the unit tests and the model pipeline.

#### Instructions

1. To Open Docker

```
open -a Docker
```

2. Build the Docker image

```
docker build -t <image_name> .
```

3. Run the unit tests and the entire model pipeline in a docker container

```
docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=<profile_name> <image_name>
```

Ensure that the AWS login is refreshed.
