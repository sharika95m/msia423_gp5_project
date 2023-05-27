import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import src.crossvalidation as cv
from sklearn.model_selection import KFold

## Successful split
def test_define_folds_success():

    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50]})
    kwargs = {'n_splits':10,'shuffle':True,'random_state':123}

    kfold = cv.define_folds(kwargs)

## Testing for incorrect data type in config
def test_define_folds_incorrect_type():
    kwargs = {'n_splits':'abc','shuffle':True,'random_state':123}

    with pytest.raises(SystemExit):
        kfold = cv.define_folds(kwargs)


## Testing for invalid value
def test_define_folds_invalid_value():
    kwargs = {'n_splits':-1,'shuffle':True,'random_state':123}

    with pytest.raises(SystemExit):
        kfold = cv.define_folds(kwargs)

## Successful get hyperparameters
def test_get_parameters_success():
    kwargs = {'DecisionTree':{'hyperparameters': {'min_samples_leaf':[3,5,10,20]}},
    'RandomForest':{'hyperparameters': {'max_features':[3,4,5],
              'min_samples_leaf':[2,3,4],
              'bootstrap':[True],
              'max_depth':[8,10,12,14,15]}}}
    
    dec_mod, rf_mod = cv.get_hyperparameters(kwargs)

## Missing keys in hyperparameters
def test_get_parameters_key_error():
    kwargs = {'DecisionTree':{'hyperparameters': {'min_samples_leaf':[3,5,10,20]}}}
    
    with pytest.raises(SystemExit):
        dec_mod, rf_mod = cv.get_hyperparameters(kwargs)

## Successful GridsearchCV DT
def test_gridsearchcv_dt_success():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'min_samples_leaf':[3,5,10,20]}
    kfold = KFold()

    dt_model = cv.gridsearchcv_dt(dt_params,kwargs,kfold,df)

## Key Error
def test_gridsearchcv_dt_keyerror():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','C']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'min_samples_leaf':[3,5,10,20]}
    kfold = KFold()

    with pytest.raises(SystemExit):
        dt_model = cv.gridsearchcv_dt(dt_params,kwargs,kfold,df)


## Invalid Hyperparameter name
def test_gridsearchcv_dt_hyper():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'invalid_hyper':[3,5,10,20]}
    kfold = KFold()

    with pytest.raises(SystemExit):
        dt_model = cv.gridsearchcv_dt(dt_params,kwargs,kfold,df)


## Invalid Hyperparameter value
def test_gridsearchcv_dt_invalid_hyper():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'min_samples_leaf':['A','B']}
    kfold = KFold()

    with pytest.raises(SystemExit):
        dt_model = cv.gridsearchcv_dt(dt_params,kwargs,kfold,df)

## More folds than samples
def test_gridsearchcv_dt_more_samples():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'min_samples_leaf':[3,5,10,20]}
    kfold = KFold(n_splits=10)

    with pytest.raises(SystemExit):
        dt_model = cv.gridsearchcv_dt(dt_params,kwargs,kfold,df)

## Successful GridsearchCV RF
def test_gridsearchcv_rf_success():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'max_features':[3,4,5],
              'min_samples_leaf':[2,3,4]}
    kfold = KFold()

    dt_model = cv.gridsearchcv_rf(dt_params,kwargs,kfold,df)

## Key Error
def test_gridsearchcv_rf_keyerror():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','C']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'max_features':[3,4,5],
              'min_samples_leaf':[2,3,4]}
    kfold = KFold()

    with pytest.raises(SystemExit):
        dt_model = cv.gridsearchcv_rf(dt_params,kwargs,kfold,df)


## Invalid Hyperparameter name
def test_gridsearchcv_rf_hyper():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'invalid_para':[3,4,5],
              'min_samples_leaf':[2,3,4]}
    kfold = KFold()

    with pytest.raises(SystemExit):
        dt_model = cv.gridsearchcv_rf(dt_params,kwargs,kfold,df)


## Invalid Hyperparameter value
def test_gridsearchcv_rf_invalid_hyper():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'max_features':['A','B'],
              'min_samples_leaf':[2,3,4]}
    kfold = KFold()

    with pytest.raises(SystemExit):
        dt_model = cv.gridsearchcv_rf(dt_params,kwargs,kfold,df)

## More folds than samples
def test_gridsearchcv_rf_more_samples():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'max_features':[3,4,5],
              'min_samples_leaf':[2,3,4]}
    kfold = KFold(n_splits=10)

    with pytest.raises(SystemExit):
        dt_model = cv.gridsearchcv_rf(dt_params,kwargs,kfold,df)

## Successful DT training
def test_train_model_dt_success():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'min_samples_leaf':3}

    dt_model = cv.train_model_dt(dt_params,df,kwargs)

## Key Error
def test_train_model_dt_keyerror():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','C']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'min_samples_leaf':3}

    with pytest.raises(SystemExit):
        dt_model = cv.train_model_dt(dt_params,df,kwargs)


## Invalid Hyperparameter name
def test_train_model_dt_hyper():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'invalid_hyper':3}

    with pytest.raises(SystemExit):
        dt_model = cv.train_model_dt(dt_params,df,kwargs)


## Invalid Hyperparameter value
def test_train_model_dt_invalid_hyper():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'min_samples_leaf':'A'}

    with pytest.raises(SystemExit):
        dt_model = cv.train_model_dt(dt_params,df,kwargs)

## Successful RF training
def test_train_model_rf_success():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    rf_params = {'max_features':3,
              'min_samples_leaf':2}

    dt_model = cv.train_model_rf(rf_params,df,kwargs)

## Key Error
def test_train_model_rf_keyerror():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','C']},'get_target':{'target_feature':'Churn_Yes'}}
    rf_params = {'max_features':3,
              'min_samples_leaf':2}

    with pytest.raises(SystemExit):
        dt_model = cv.train_model_rf(rf_params,df,kwargs)


## Invalid Hyperparameter name
def test_train_model_rf_hyper():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    rf_params = {'invalid_para':3,
              'min_samples_leaf':2}

    with pytest.raises(SystemExit):
        dt_model = cv.train_model_rf(rf_params,df,kwargs)


## Invalid Hyperparameter value
def test_train_model_rf_invalid_hyper():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    rf_params = {'max_features':'A',
              'min_samples_leaf':2}

    with pytest.raises(SystemExit):
        dt_model = cv.train_model_rf(rf_params,df,kwargs)

## Invalid save path Decision Tree
def test_save_model_dt():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    dt_params = {'min_samples_leaf':3}

    dt_model = cv.train_model_dt(dt_params,df,kwargs)
    save_path = Path('/random_path/check.csv')

    with pytest.raises(SystemExit):
        cv.save_model(dt_model,save_path)

## Invalid save path Decision Tree
def test_save_model_rf():
    df = pd.DataFrame({"A":[1,2,3,4,5],"B":[10,20,30,40,50],'Churn_Yes':[1,0,1,0,0]})
    kwargs = {'get_features':{'features_to_use':['A','B']},'get_target':{'target_feature':'Churn_Yes'}}
    rf_params = {'max_features':3,
              'min_samples_leaf':2}

    dt_model = cv.train_model_rf(rf_params,df,kwargs)
    save_path = Path('/random_path/check.csv')

    with pytest.raises(SystemExit):
        cv.save_model(dt_model,save_path)


