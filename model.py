#Importing required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.utils import resample,shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
from random import sample
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hi'