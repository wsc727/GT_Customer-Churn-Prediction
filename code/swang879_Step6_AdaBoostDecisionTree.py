# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 21:32:37 2022

@author: chelsea.wang
"""

"""   
This script is aimed to explore AdaBoost Decision Tree model for both datasets
MUST complete the following requirements before running this script
1. Run Script for Step1 which contains all required modules and functions
2. Download the Cleaned data files and save them in the same folder with all scripts
3. Navigate to the folder with Cleaned data files
"""

""" ABDT Prep FOR Telcom"""
# according to ref in dict_excel, KNN not need feature normlization

#Convert Catgorical Var to Dummy variable, then split to Train test for Xs & Y
df_f = pd.read_csv("Telco_final.csv")
y_col= ["Churn"]
other_col = ["set_type"]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
dummy_cols= ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']

#dummy 
df_temp = pd.get_dummies(df_f[dummy_cols])

### teclcom only, too many "no internet serice" duplicateed cols ###
#check and clean
lst_noint = []
for lst in df_temp.columns:
    if "No internet service" in lst: lst_noint.append(lst)
temp_sum = df_temp[lst_noint].sum(axis = 1)
sum(temp_sum.apply(lambda x: 0 if x in [0, 6] else 1)) # 0
#SAFE TO REMOVE
dummy_col_clean = list(set(df_temp.columns) - set(lst_noint))
df_temp_clean = df_temp[dummy_col_clean]
df_temp_clean ["NoInternetService"] = df_temp[lst_noint[0]]

#Train Test Split    
df_ml = pd.concat([df_f[y_col +other_col +num_cols],df_temp_clean],1)


#final df for training and testing

X_train = df_ml[df_ml["set_type"] == "train"].drop(y_col + other_col, axis=1) 
Y_train =df_ml[df_ml["set_type"] == "train"][y_col[0]]

x_test = df_ml[df_ml["set_type"] == "test"].drop(y_col + other_col, axis=1) 
y_test =df_ml[df_ml["set_type"] == "test"][y_col[0]]


""" ABDT Prep FOR Bank data SKIP when analyzing Telcom """
# according to ref in dict_excel, KNN not need feature normlization

#Convert Catgorical Var to Dummy variable, then split to Train test for Xs & Y
df_f = pd.read_csv("Bank_final.csv")
y_col= ["Exited"]
other_col = ["set_type"]
num_cols = ["CreditScore", 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
dummy_cols= ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

#dummy 
df_temp = pd.get_dummies(df_f[dummy_cols])

#Train Test Split    
df_ml = pd.concat([df_f[y_col +other_col+num_cols],df_temp],1)

#final df for training and testing

X_train = df_ml[df_ml["set_type"] == "train"].drop(y_col + other_col, axis=1) 
Y_train =df_ml[df_ml["set_type"] == "train"][y_col[0]]

x_test = df_ml[df_ml["set_type"] == "test"].drop(y_col + other_col, axis=1) 
y_test =df_ml[df_ml["set_type"] == "test"][y_col[0]]

""" ABDT Training for Telcom"""

""" Adjust nestimator """
# Previous Tecl alpga = 0.0008257497568358967 
basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha= 0.0008257497568358967)

model =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R")

#model =  AdaBoostClassifier(base_estimator=basemodel, learning_rate=learning_rate, n_estimators=n_estimators,algorithm="SAMME.R")
n_estimators = [1, 5, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "n_estimators", param_range = n_estimators, cv = 10)

title = "Validation Curve for AdaBoosted Descrsion tree"
x_var="n_estimators"
x_range = n_estimators  
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
# Best is 1, 0.7941910321740134

""" Adjust learning rate (n_estimator = 50)"""
basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha= 0.0008257497568358967)

model =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R")
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "learning_rate", param_range = learning_rates, cv = 10)

title = "Validation Curve for AdaBoosted Descrsion tree"
x_var="learning rate" 
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
# {0.1: 0.7659476574481768}

""" Adjust alpha (n_estimator = 50, learning = 0.1)"""
ccp_alphas = [2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4]
train_scores = pd.DataFrame(columns = ccp_alphas)
valid_scores= pd.DataFrame(columns = ccp_alphas)

for alpha in ccp_alphas:
    basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha = alpha)

    model_temp =  AdaBoostClassifier(base_estimator=basemodel, learning_rate=0.1, algorithm="SAMME.R")
    train_scores_temp, valid_scores_temp = validation_curve(model_temp, X_train, Y_train, cv = 10, param_name = "n_estimators", param_range = [50])
    train_scores[alpha] = list(train_scores_temp[0])
    valid_scores[alpha] = list(valid_scores_temp[0])


title = "Validation Curve for AdaBoosted Descrsion tree"
x_var="ccp_alphas"
 
df_validation_curve_plt(train_scores, valid_scores, x_var, title)

df_best_param_accuracy(valid_scores) # {0.0004: 0.7702184237866719}

"""  Gridsearch  """

param_grid = {
    'n_estimators': [1, 5, 10, 25, 50],
    'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25]
}

basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha= 0.004)
model =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R")

# Instantiate the grid search model # 10-fold cv
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 10)

# Fit the grid search to the data
grid_search.fit(X_train, Y_train)

grid_search.best_params_ #n_estimators= 25, learning_rate = 0.1, ccp_alpha= 0.002
grid_search.best_score_ 

"""  learning curve Telcom""" 
#
title = "Learning Curves (AdaBoosted Decision tree)"
basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha= 0.002)
estimator =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R", n_estimators= 25, learning_rate = 0.1)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()


"""  BDT FINAL PARAMS Telcom """
basemodel = tree.DecisionTreeClassifier(ccp_alpha = 0.002, random_state=123)
clf =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R", n_estimators= 25, learning_rate = 0.1)
clf.fit(X_train, Y_train)

clf.score(X_train, Y_train) # 0.8122714343762698
clf.score(x_test, y_test) # 0.804739336492891

Y_pred = clf.predict(X_train)
confusion_matrix(Y_train, Y_pred)

y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred)


""" ABDT Training for Bank"""

""" Adjust nestimator """
# Previous Tecl alpga = 0.0020341898371013523 
basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha= 0.0020341898371013523)
model =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R")

n_estimators = [1, 5, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "n_estimators", param_range = n_estimators, cv = 10)

title = "Validation Curve for AdaBoosted Descrsion tree"
x_var="n_estimators"
x_range = n_estimators  
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) # Best is 5: 0.8541236460249337

""" Adjust learning rate (n_estimator = 5)"""
model =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R", n_estimators = 5)
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "learning_rate", param_range = learning_rates, cv = 10)

title = "Validation Curve for AdaBoosted Descrsion tree"
x_var="learning rate" 
x_range = learning_rates
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) # {0.1: 0.7659476574481768}

""" Adjust alpha (n_estimator = 50, learning = 0.1)"""

ccp_alphas = [1e-4, 4e-4, 8e-4, 1e-3, 2e-3, 4e-3, 8e-3]
train_scores = pd.DataFrame(columns = ccp_alphas)
valid_scores= pd.DataFrame(columns = ccp_alphas)

for alpha in ccp_alphas:
    basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha = alpha)

    model_temp =  AdaBoostClassifier(base_estimator=basemodel, learning_rate=0.8, algorithm="SAMME.R")
    train_scores_temp, valid_scores_temp = validation_curve(model_temp, X_train, Y_train, cv = 10, param_name = "n_estimators", param_range = [5])
    train_scores[alpha] = list(train_scores_temp[0])
    valid_scores[alpha] = list(valid_scores_temp[0])

title = "Validation Curve for AdaBoosted Descrsion tree"
x_var="ccp_alphas"
 
df_validation_curve_plt(train_scores, valid_scores, x_var, title)

df_best_param_accuracy(valid_scores) # 0.002: 0.8596946658491722

"""  Gridsearch  """
# NOTE ccp_alpha value needs to be manually changed below to find the best n_est & lr for each ccp_alpha
# ccp_alpha= 0020341898371013523 has the best performance and is the only one listed below
param_grid = {
    'n_estimators': [5, 15, 25, 55, 75],
    'learning_rate': [ 0.5, 0.6, 0.75, 0.8, 0.85]
}

basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha= 0020341898371013523)
model =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R")

# Instantiate the grid search model # 10-fold cv
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 10)

# Fit the grid search to the data
grid_search.fit(X_train, Y_train)

grid_search.best_params_ # ccp_alpha= 0.0020341898371013523, n_estimators= 15, learning_rate = 0.5
grid_search.best_score_ 

"""  learning curve Bank""" 
title = "Learning Curves (AdaBoosted Decision tree)"
basemodel = tree.DecisionTreeClassifier(random_state=123, ccp_alpha= 0.0020341898371013523)
estimator =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R", n_estimators= 15, learning_rate = 0.5)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()


"""  BDT FINAL PARAMS Bank """
basemodel = tree.DecisionTreeClassifier(ccp_alpha = 0.0020341898371013523 , random_state=123)
clf =  AdaBoostClassifier(base_estimator=basemodel, algorithm="SAMME.R", n_estimators= 15, learning_rate = 0.5)
clf.fit(X_train, Y_train)

clf.score(X_train, Y_train) 
clf.score(x_test, y_test) 

Y_pred = clf.predict(X_train)
confusion_matrix(Y_train, Y_pred)

y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred)





