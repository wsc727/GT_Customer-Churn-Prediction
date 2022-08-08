# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:27:42 2022

@author: chelsea.wang
"""

import os
from math import log10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve

from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import graphviz

#check module version
#from importlib.metadata import version
# version(pandas) # no need to import


""" check and drop missing values """

def missing_value_idx(df, col_lst):
    missing_idx = []
    for col in col_lst:
        cond1 =df.index[df[col].isna() == True].tolist() # sum also works
        cond2 = df.index[df[col] == " "].tolist()
        if len(cond1) > 0 : 
            output = {col: cond1}
            missing_idx.append(output)
        if len(cond2) > 0 : 
            output = {col: cond2}
            missing_idx.append(output)
    return missing_idx

"""  fix col data type & generate catg & cont lists for later ML""" 

def var_type_fix (df, cont_lst):
    cat_lst = [var for var in col_lst if var not in cont_lst] # set(a) - set (b). Not a-b
    df[cont_lst] = df[cont_lst].astype("float")
    df[cat_lst] = df[cat_lst].astype("str")  #df[col].dtype.name == "xyz"
    return df

""" Train test split with ratio of churn/non-churn fixed, return cleaned df  """

def train_test_split(df, y, pos_str, neg_str):
    df_ng = df[df[y] == neg_str] 
    df_pos = df[df[y] == pos_str] 

    ng_tr, ng_ts = np.split(df_ng.sample(frac=1, random_state=123), [int(.7*len(df_ng))])
    ng_tr["set_type"] = "train"
    ng_ts["set_type"] = "test"
    
    ps_tr, ps_ts = np.split(df_pos.sample(frac=1, random_state=123), [int(.7*len(df_pos))])
    ps_tr["set_type"] = "train"
    ps_ts["set_type"] = "test"
    #generate final df
    df_f = pd.DataFrame(columns= ng_tr.columns)
    df_lst = [ps_tr, ps_ts, ng_tr, ng_ts]
    for i in df_lst:
        df_f = df_f.append(i, ignore_index=True)
    return df_f

""" plot validation curve  for KNN, SVM, DT """
def validation_curve_plt (train_scores, valid_scores, x_var, x_range, title):
    ## https://scikit-learn.org/stable/modules/learning_curve.html
    ## https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
    
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.title(title)
    plt.xlabel(x_var)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(x_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(x_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.2, color="darkorange", lw=lw)

    plt.plot(x_range, valid_scores_mean, label="Cross-validation score", color="navy", lw=lw)

    plt.fill_between(x_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std,
                     alpha=0.2, color="navy", lw=lw)
    plt.legend(loc="lower left")
    return plt.show()

""" plot validation curve  for ABDT  """
def df_validation_curve_plt (train_scores, valid_scores, x_var, title):

    train_scores_mean = train_scores.apply(lambda x: np.mean(x)) #series, index is alpha, 
    train_scores_std = train_scores.apply(lambda x: np.std(x))
    valid_scores_mean = valid_scores.apply(lambda x: np.mean(x))
    valid_scores_std = valid_scores.apply(lambda x: np.std(x))
    
    plt.title(title)
    x_range = list(train_scores_mean.index)
    plt.xlabel(x_var)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(x_range, train_scores_mean.to_numpy(dtype=object), label="Training score", color="darkorange", lw=lw)
    plt.fill_between(x_range, list(train_scores_mean.to_numpy(dtype=object) - train_scores_std.to_numpy(dtype=object)), 
                     list(train_scores_mean.to_numpy(dtype=object) + train_scores_std.to_numpy(dtype=object)),
                     alpha=0.2, color="darkorange", lw=lw)

    plt.plot(x_range, valid_scores_mean.to_numpy(dtype=object), label="Cross-validation score", color="navy", lw=lw)

    plt.fill_between(x_range, list(valid_scores_mean.to_numpy(dtype=object) - valid_scores_std.to_numpy(dtype=object)), 
                     list(valid_scores_mean.to_numpy(dtype=object) + valid_scores_std.to_numpy(dtype=object)),
                     alpha=0.2, color="navy", lw=lw)
    plt.legend(loc="lower left")
    return plt.show()

""" return parameters with highest validation accuracy in validation curve for KNN, SVM, DT """
def best_param_accuracy (valid_scores, x_range):

    temp_array = np.mean(valid_scores, axis=1)
    max_score = max(temp_array)
    max_idx = np.argmax(temp_array)
    return {x_range[max_idx]:max_score}

""" return parameters with highest validation accuracy in validation curve for ABDT """
def df_best_param_accuracy (valid_scores):

    temp = valid_scores.apply(lambda x:np.mean(x))
    max_score = temp.apply(lambda x:np.max(x)).max()
    max_idx = temp.apply(lambda x:np.max(x)).idxmax()
    return {max_idx:max_score}

""" plot learning curve for KNN, SVM, DT, ADBT """
### https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html  
def learning_curve_plt(title,estimator, X_train, Y_train):
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))


    axes[0].set_title(title)
    ylim=(0, 1.01)
    if ylim is not None:
       axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes=np.linspace(0.1, 1.0, 10)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X_train, Y_train, cv=10, train_sizes=train_sizes, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

# Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    axes[0].legend(loc="lower right")

# Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(train_sizes,fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    return plt

""" plot learning curve for NN """
def NN_learning_curve(history, title, epoch=200, y_max = 1):

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,6))
    plt.ylim(0.0, y_max+0.1)
    plt.suptitle(title, size=14)
     ## training    
    x_max = epoch+1
    ax[0].set(title="Training")      
    ax[0].plot(np.arange(1, x_max), history.history['loss'], label='Loss', color='black')
    ax[0].plot(np.arange(1, x_max), history.history['accuracy'], label='accuracy')
    ax[0].plot(np.arange(1, x_max), history.history['precision'], label='Precision')  
    ax[0].plot(np.arange(1, x_max), history.history['recall'], label='Recall')
    ax[0].legend(loc = 'lower left')
    ax[0].set_xlabel('Epoch', size=12)
        
    ## validation    
    ax[1].set(title="Validation")     
    ax[1].plot(np.arange(1, x_max), history.history['val_loss'], label='loss', color='black')    
    ax[1].plot(np.arange(1, x_max), history.history['val_accuracy'], label='accuracy')
    ax[1].plot(np.arange(1, x_max), history.history['val_precision'], label='Precision')  
    ax[1].plot(np.arange(1, x_max), history.history['val_recall'], label='Recall')
    ax[1].legend(loc = 'lower left')
    ax[1].set_xlabel('Epoch', size=12)
    return plt

""" Training time for NN Tensor flow """
import time
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.process_time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch,time.process_time() - self.timetaken))
    def on_train_end(self,logs = {}):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.show()