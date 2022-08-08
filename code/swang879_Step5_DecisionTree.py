# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:44:59 2022

@author: chelsea.wang
"""

"""   
This script is aimed to explore Decision Tree model for both datasets
MUST complete the following requirements before running this script
1. Run Script for Step1 which contains all required modules and functions
2. Download the Cleaned data files and save them in the same folder with all scripts
3. Navigate to the folder with Cleaned data files
"""

""" DT Prep FOR Telcom"""

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


""" DT Prep FOR Bank data SKIP when analyzing Telcom """
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

""" DT Training, SAME for Both DATA """
#Tree refs
#pruning code: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
# pruning code 2: https://ranvir.xyz/blog/practical-approach-to-tree-pruning-using-sklearn/
# DT sturcture codel & explain: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
# DT code: https://scikit-learn.org/stable/modules/tree.html
# cost complexity pruning explain: http://mlwiki.org/index.php/Cost-Complexity_Pruning
# cost complexity pruning explain 2: https://online.stat.psu.edu/stat508/lesson/11/11.8/11.8.2
# IG vs complexity-cost: IG seems to be pre-pruning tech vs. comst: pst pruning based on previous ref:
    ##https://towardsdatascience.com/decision-tree-classifier-and-cost-computation-pruning-using-python-b93a0985ea77 


""" Pruning  USE default Gini"""
model = tree.DecisionTreeClassifier(random_state=123)
path = model.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas = path.ccp_alphas

train_scores, valid_scores = validation_curve(model, X_train, Y_train, param_name = "ccp_alpha", param_range = ccp_alphas, cv = 10)

title = "Validation Curve for Descrsion tree (accuracy vs.alpha)"
x_var="ccp_alpha"
x_range = ccp_alphas  
validation_curve_plt(train_scores, valid_scores, x_var, x_range, title)

best_param_accuracy(valid_scores, x_range) 
#Telcom 0.0008257497568358967 
#bank 0.0020341898371013523: 0.8534075209482935

"""  No pruning situation  """
model_np = tree.DecisionTreeClassifier(random_state=123, ccp_alpha=ccp_alphas[-1])
model_np.fit(X_train,Y_train)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
       model_np.tree_.node_count, ccp_alphas[-1]
    )
)

"""  learning curve Telcom""" 
title = "Learning Curves (Decision tree)"
estimator = tree.DecisionTreeClassifier(ccp_alpha = 0.0008257497568358967, random_state=123)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()

"""  learning curve Bank""" 
title = "Learning Curves (Decision tree)"
estimator = tree.DecisionTreeClassifier(ccp_alpha = 0.0020341898371013523, random_state=123)
learning_curve_plt(title,estimator, X_train, Y_train)
plt.show()

"""  DT FINAL PARAMS Telcom """
clf = tree.DecisionTreeClassifier(ccp_alpha = 0.0008257497568358967, random_state=123)
clf.fit(X_train, Y_train)

clf.score(X_train, Y_train) # 0.8037383177570093
clf.score(x_test, y_test) # 0.7971563981042654

Y_pred = clf.predict(X_train)
confusion_matrix(Y_train, Y_pred)

y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred)

"""  DT FINAL PARAMS Bank """
clf = tree.DecisionTreeClassifier(ccp_alpha = 0.0020341898371013523, random_state=123)
clf.fit(X_train, Y_train)

clf.score(X_train, Y_train) # 0.8542648949849978
clf.score(x_test, y_test) # 0.8533822059313562

Y_pred = clf.predict(X_train)
confusion_matrix(Y_train, Y_pred)

y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred)

"""  Tree Visulization """
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X_train.columns)
graph = graphviz.Source(dot_data)  
graph






