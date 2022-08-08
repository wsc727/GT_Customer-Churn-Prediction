# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:20:18 2022

@author: chelsea.wang
"""


"""   
This script is aimed to explore KNN model for both datasets
MUST complete the following requirements before running this script
1. Run Script for Step1 which contains all required modules and functions
2. Download the Cleaned data files and save them in the same folder with all scripts
3. Navigate to the folder with Cleaned data files
"""


""" KNN Prep FOR Telcom data"""
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
X_train = df_ml[df_ml["set_type"] == "train"].drop(y_col + other_col, axis=1) 
Y_train =df_ml[df_ml["set_type"] == "train"][y_col[0]]

x_test = df_ml[df_ml["set_type"] == "test"].drop(y_col + other_col, axis=1) 
y_test =df_ml[df_ml["set_type"] == "test"][y_col[0]]


""" KNN Prep FOR Bank data SKIP when analyzing Telcom """

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

X_train = df_ml[df_ml["set_type"] == "train"].drop(y_col + other_col, axis=1) 
Y_train =df_ml[df_ml["set_type"] == "train"][y_col[0]]

x_test = df_ml[df_ml["set_type"] == "test"].drop(y_col + other_col, axis=1) 
y_test =df_ml[df_ml["set_type"] == "test"][y_col[0]]


""" KNN Training, SAME for Both DATA """

#KNN  Uniform
model = KNeighborsClassifier(weights = 'uniform')
train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "n_neighbors", param_range = list(np.arange(5, 105, 5)), cv = 10)

title = "Validation Curve for KNN (uniform)"
x_var="n_neighbors"
x_range = list(np.arange(5, 105, 5))
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
#telcom {70: 0.791348389650225}  
#bank {5: 0.8101126098508074}

#KNN  weighted
model = KNeighborsClassifier(weights = 'distance')
train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "n_neighbors", param_range = list(np.arange(5, 105, 5)), cv = 10)

title = "Validation Curve for KNN (weighted)"
x_var="n_neighbors"
x_range = list(np.arange(5, 105, 5))
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range)  
# telcom {90: 0.7769199689968503} 
# bank {5: 0.8119701614551401}

### KNN unifrmn p = 1 manhattan_distance

#KNN  Uniform Mahaton 
model = KNeighborsClassifier(weights = 'uniform', p =1)
train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "n_neighbors", param_range = list(np.arange(5, 105, 5)), cv = 10)

title = "Validation Curve for KNN (uniform, manhattan_distance)"
x_var="n_neighbors"
x_range = list(np.arange(5, 105, 5))
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
#{70: 0.7897231979419186}  
#bank {15: 0.8146877171469447}

#KNN  weighted
model = KNeighborsClassifier(weights = 'distance', p =1)
train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "n_neighbors", param_range = list(np.arange(5, 105, 5)), cv = 10)

title = "Validation Curve for KNN (weighted, manhattan_distance)"
x_var="n_neighbors"
x_range = list(np.arange(5, 105, 5))
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
#Telcom {85: 0.7771223964775145} 
#bank {15: 0.8181169016963008}

"""  lEARNING Curve for similar performance models"""

"""  Telcom """ 
title = "Learning Curves (KNN, uniform, k=70, p=1)"
estimator = KNeighborsClassifier(weights = 'uniform', n_neighbors = 70, p=1)
learning_curve_plt(title,estimator,X_train, Y_train)
plt.show()

title = "Learning Curves (KNN, uniform, k=70, p=2)"
estimator = KNeighborsClassifier(weights = 'uniform', n_neighbors = 70, p=2)
learning_curve_plt(title,estimator,X_train, Y_train)
plt.show()

"""  Bank """ 
title = "Learning Curves (KNN, uniform, k=15, p=1)"
estimator = KNeighborsClassifier(weights = 'uniform', n_neighbors = 15, p=1)
learning_curve_plt(title,estimator,X_train, Y_train)
plt.show()

title = "Learning Curves (KNN, uniform, k=5, p=2)"
estimator = KNeighborsClassifier(weights = 'uniform', n_neighbors = 5, p=2)
learning_curve_plt(title,estimator,X_train, Y_train)
plt.show()

"""  Telcom: KNN FINAL PARAMS : K = 70, weights  = uniform, p =1 """
knn = KNeighborsClassifier(weights = 'uniform', n_neighbors = 70, p=1)
knn.fit(X_train, Y_train)

knn.score(X_train, Y_train)  #scaled 0.7945956928078017
knn.score(x_test, y_test) #scaled 0.7914691943127962

Y_pred = knn.predict(X_train)
confusion_matrix(Y_train, Y_pred)

y_pred = knn.predict(x_test)
confusion_matrix(y_test, y_pred)


"""  DATA 2: KNN FINAL PARAMS : K = 15, weights  = uniform, p =1 """
knn = KNeighborsClassifier(weights = 'uniform', n_neighbors = 15, p=1)
knn.fit(X_train, Y_train)

knn.score(X_train, Y_train) #scaled 	0.8298328332618946
knn.score(x_test, y_test) #scaled 0.8153948683772076

Y_pred = knn.predict(X_train)
confusion_matrix(Y_train, Y_pred)

y_pred = knn.predict(x_test)
confusion_matrix(y_test, y_pred)













