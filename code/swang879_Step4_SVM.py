# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:44:59 2022

@author: chelsea.wang
"""

"""   
This script is aimed to explore SVM model for both datasets
MUST complete the following requirements before running this script
1. Run Script for Step1 which contains all required modules and functions
2. Download the Cleaned data files and save them in the same folder with all scripts
3. Navigate to the folder with Cleaned data files
"""


""" SVM Prep FOR Telcom data"""

df_f = pd.read_csv("Telco_final.csv")
y_col= ["Churn"]
other_col = ["set_type"]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
dummy_cols= ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']

# retrieve just the numeric input values
# Normalization to [0,1] due to dummy variables
scaler = MinMaxScaler()
scaled_num = scaler.fit_transform(df_f[num_cols])
temp_scale = pd.DataFrame(scaled_num)
temp_scale.columns = num_cols

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
df_ml = pd.concat([df_f[y_col +other_col],df_temp_clean, temp_scale],1)

#final df for training and testing
# select 20% for params tuning
df_tune_temp = df_ml[df_ml["set_type"] == "train"]
df_tune = np.split(df_tune_temp.sample(frac=1, random_state=123), [int(.2*len(df_ml))])[0]

X_tune = df_tune.drop(y_col + other_col, axis=1)
Y_tune = df_tune[y_col[0]]

X_train = df_ml[df_ml["set_type"] == "train"].drop(y_col + other_col, axis=1) 
Y_train =df_ml[df_ml["set_type"] == "train"][y_col[0]]

x_test = df_ml[df_ml["set_type"] == "test"].drop(y_col + other_col, axis=1) 
y_test =df_ml[df_ml["set_type"] == "test"][y_col[0]]

# Only use 20% for params tuning, too slow for linear kernel


""" SVM Prep FOR Bank data SKIP when analyzing Telcom """
# according to ref in dict_excel, KNN not need feature normlization

#Convert Catgorical Var to Dummy variable, then split to Train test for Xs & Y
df_f = pd.read_csv("Bank_final.csv")
y_col= ["Exited"]
other_col = ["set_type"]
num_cols = ["CreditScore", 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
dummy_cols= ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

# Normalization to [0,1] due to dummy variables
scaler = MinMaxScaler()
scaled_num = scaler.fit_transform(df_f[num_cols])
temp_scale = pd.DataFrame(scaled_num)
temp_scale.columns = num_cols

#dummy 
df_temp = pd.get_dummies(df_f[dummy_cols])

#Train Test Split    
df_ml = pd.concat([df_f[y_col +other_col],df_temp, temp_scale],1)

#final df for training and testing
# select 20% for params tuning
df_tune_temp = df_ml[df_ml["set_type"] == "train"]
df_tune = np.split(df_tune_temp.sample(frac=1, random_state=123), [int(.1*len(df_ml))])[0]

X_tune = df_tune.drop(y_col + other_col, axis=1)
Y_tune = df_tune[y_col[0]]

X_train = df_ml[df_ml["set_type"] == "train"].drop(y_col + other_col, axis=1) 
Y_train =df_ml[df_ml["set_type"] == "train"][y_col[0]]

x_test = df_ml[df_ml["set_type"] == "test"].drop(y_col + other_col, axis=1) 
y_test =df_ml[df_ml["set_type"] == "test"][y_col[0]]

""" SVM Training, SAME for Both DATA """
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html 

""" Linear kernel """
model = svm.SVC(kernel='linear', random_state=123)
train_scores, valid_scores = validation_curve(model, X_tune, Y_tune, param_name = "C", param_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3] , cv = 10)

title = "Validation Curve for SVM (linear kernel)"
x_var="C (on log10 scale)"

x_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]   
x_range = [log10(x) for x in x_range]
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
# Telcom {1, 0.7951621073961499} 
#bank 1e-5.0: 0.8019999999999999

"""  learning curve Telcom""" 
title = "Learning Curves (SVM_linear, C=1)"
estimator = svm.SVC(kernel='linear', C=1, random_state=123)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()

"""  learning curve Bank """ 
title = "Learning Curves (SVM_linear, C=1e-5)"
estimator = svm.SVC(kernel='linear', C=1e-5, random_state=123)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()

""" rbf kernel """
#C
model = svm.SVC(kernel='rbf', random_state=123)
train_scores, valid_scores = validation_curve(model, X_tune, Y_tune, param_name = "C", param_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3], cv = 10)

title = "Validation Curve for SVM (rbf kernel)"
x_var="C (on log10 scale)"

x_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]   
x_range = [log10(x) for x in x_range]
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
# Telcom: C=1, 0.7909371833839918
 #bank: 10, 0.8220000000000001

# Use best C, change gamma
model = svm.SVC(kernel='rbf', C = 10, random_state=123)
train_scores, valid_scores = validation_curve(model, X_tune, Y_tune, param_name = "gamma", param_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3], cv = 10)

title = "Validation Curve for SVM (rbf kernel)"
x_var="Gamma (on log10 scale)"

x_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]   
x_range = [log10(x) for x in x_range]
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
#Telcom gamma=1e-2, 0.7987436676798378 
#bank gamma = 1, 0.8109999999999999

"""  learning curve Telcom""" 
title = "Learning Curves (SVM_rbf, C=1, gamma = 0.01)"
estimator = svm.SVC(kernel='rbf', C=1, gamma=0.01,  random_state=123)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()

"""  learning curve BANK""" 
title = "Learning Curves (SVM_rbf, C=10, gamma = 1)"
estimator = svm.SVC(kernel='rbf', C=10, gamma=1,  random_state=123)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()

""" polynomial kernel """
#C
model = svm.SVC(kernel='poly', random_state=123)
train_scores, valid_scores = validation_curve(model, X_tune, Y_tune, param_name = "C", param_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3], cv = 10)

title = "Validation Curve for SVM (poly kernel, degree = 3)"
x_var="C (on log10 scale)"

x_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]   
x_range = [log10(x) for x in x_range]
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
#Telcom C=1, 0.7958966565349543 
#bank C=1, 0.825

# Use best C, change gamma
model = svm.SVC(kernel='poly', C = 1, random_state=123)
train_scores, valid_scores = validation_curve(model, X_tune, Y_tune, param_name = "gamma", param_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3], cv = 10)

title = "Validation Curve for SVM (poly kernel, degree = 3)"
x_var="Gamma (on log10 scale)"

x_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]   
x_range = [log10(x) for x in x_range]
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) # gamma=0.1, 0.7951975683890576 #bank gamma = 1, 0.825

"""  learning curve Telcom""" 
title = "Learning Curves (SVM_poly, C=1, gamma = 0.1, degree=3)"
estimator = svm.SVC(kernel='poly', C=1, gamma=0.1,  random_state=123)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()

"""  learning curve Bank""" 
title = "Learning Curves (SVM_poly, C=1, gamma = 1, degree=3)"
estimator = svm.SVC(kernel='poly', C=1, gamma=1,  random_state=123)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()

"""  SVM FINAL PARAMS linear Telcom """
clf = svm.SVC(kernel='linear', C=1, random_state=123)
clf.fit(X_train, Y_train)

clf.score(X_train, Y_train) # 0.7988622511174319
clf.score(x_test, y_test) # 0.795260663507109

Y_pred = clf.predict(X_train)
confusion_matrix(Y_train, Y_pred)

y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred)


"""  SVM FINAL PARAMS poly Bank"""
clf = svm.SVC(kernel='poly', C=1, gamma=1, random_state=123)
clf.fit(X_train, Y_train)

clf.score(X_train, Y_train) # 0.8611230175739392
clf.score(x_test, y_test) # 0.8607130956347884

Y_pred = clf.predict(X_train)
confusion_matrix(Y_train, Y_pred)

y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred)









