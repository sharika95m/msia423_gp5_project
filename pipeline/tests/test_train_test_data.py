import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import src.train_test_data as ttd

## Normal train_test_split
def test_train_test_divide_success():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50]})
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 42},
              'upsample_train':{'random_state':20,'axis':0,'ignore_index':True,'sort':False}}
    
    train,test = ttd.train_test_data_divide(df,kwargs)

    assert( (train.shape[0]==4) & (test.shape[0]==1))

## Missing split_data key
def test_train_test_divide_missing_key():

        df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50]})
        kwargs = {'upsample_train':{'random_state':20,'axis':0,'ignore_index':True,'sort':False}}

        with pytest.raises(SystemExit):
            train,test = ttd.train_test_data_divide(df,kwargs)

## Incorrect train_test_split ratio
def test_train_test_divide_incorrect_split():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50]})
    kwargs = {'split_data': {'test_size': 1.2, 'random_state': 42},
              'upsample_train':{'random_state':20,'axis':0,'ignore_index':True,'sort':False}}
    
    with pytest.raises(SystemExit):
        train,test = ttd.train_test_data_divide(df,kwargs)

## Missing dataframe
def test_train_test_divide_miss_df():

    df = pd.DataFrame()
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 42},
              'upsample_train':{'random_state':20,'axis':0,'ignore_index':True,'sort':False}}
    
    with pytest.raises(SystemExit):
        train,test = ttd.train_test_data_divide(df,kwargs)


## Incompatible data types
def test_train_test_divide_alpha_split():

    df = pd.DataFrame()
    kwargs = {'split_data': {'test_size': 'abc', 'random_state': 42},
              'upsample_train':{'random_state':20,'axis':0,'ignore_index':True,'sort':False}}
    
    with pytest.raises(SystemExit):
        train,test = ttd.train_test_data_divide(df,kwargs)


## Incompatible data types - 2
def test_train_test_divide_alpha_split_2():

    df = pd.DataFrame()
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 'abc'},
              'upsample_train':{'random_state':20,'axis':0,'ignore_index':True,'sort':False}}
    
    with pytest.raises(SystemExit):
        train,test = ttd.train_test_data_divide(df,kwargs)


## Successful upsampling
def test_train_test_divide_upsample_success():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,0,0,0]})
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 42},
              'upsample_train':{'random_state':20,'axis':0,'ignore_index':True,'sort':False}}
    
    upsample_df = ttd.upsample_train(df,kwargs)

    df_true = pd.DataFrame({"A":[1,1,1,1,2,3,4,5],"B":[10,10,10,10,20,30,40,50],'Churn_Yes':[1,1,1,1,0,0,0,0]})

    assert df_true.equals(upsample_df)

## Incorrect axis
def test_train_test_divide_incorrect_axis():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,0,0,0]})
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 42},
              'upsample_train':{'random_state':20,'axis':1,'ignore_index':True,'sort':False}}
    
    upsample_df = ttd.upsample_train(df,kwargs)

    assert df.equals(upsample_df)


## Incorrect parameter type 
def test_train_test_divide_incorrect_type():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,0,0,0]})
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 42},
              'upsample_train':{'random_state':'abc','axis':0,'ignore_index':True,'sort':True}}
    
    upsample_df = ttd.upsample_train(df,kwargs)

    assert df.equals(upsample_df)

## Incorrect parameter type 2
def test_train_test_divide_incorrect_type_2():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,0,0,0]})
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 42},
              'upsample_train':{'random_state':20,'axis':'abc','ignore_index':True,'sort':True}}
    
    upsample_df = ttd.upsample_train(df,kwargs)

    assert df.equals(upsample_df)

## Missing Key
def test_train_test_divide_missing_key():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,0,0,0]})
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 42}}
    
    upsample_df = ttd.upsample_train(df,kwargs)

    assert df.equals(upsample_df)

## No Minority
def test_train_test_divide_no_minority():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[0,0,0,0,0]})
    kwargs = {'split_data': {'test_size': 0.2, 'random_state': 42},
              'upsample_train':{'random_state':20,'axis':1,'ignore_index':True,'sort':True}}
    
    upsample_df = ttd.upsample_train(df,kwargs)

    assert df.equals(upsample_df)


## Invalid save path
def test_save_data_invalid():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[0,0,0,0,0]})
    df_1 = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[0,0,0,0,0]})
    df_2 = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[0,0,0,0,0]})
    save_path = Path('/random_path/check.csv')

    with pytest.raises(SystemExit):
        ttd.save_data(df,df_1,df_2,save_path)




