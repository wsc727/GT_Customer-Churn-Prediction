# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:03:29 2022

@author: chelsea.wang
"""

"""   
This script is aimed to clean raw dataset
MUST complete the following requirements before running this script
1. Run Script for Step1 which contains all required modules and functions
2. Download the raw data sets and save them in the same folder with all scripts
3. Change file_loc =  "/Users\\xyz\\Desktop\\ML stuffs\\ML" to corresponding file location 
"""


""" Data 1 """
file_loc = "/Users\\xyz\\Desktop\\ML stuffs\\ML"
file_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
cont_lst = ["tenure", "MonthlyCharges", "TotalCharges"]

os.chdir(file_loc)
df = pd.read_csv(file_name)
col_lst = list(df.columns)

missing_value_idx(df, col_lst)    
drop_row_idx = [488, 753, 936, 1082, 1340, 3331, 3826, 4380, 5218, 6670, 6754]
df = df.drop(drop_row_idx)

df_fix = var_type_fix (df, cont_lst)

#ID uniqueness check
unq_var = "customerID"
temp_len = len(df_fix[unq_var].unique())
if  temp_len == len(df_fix): # df[unq_var].unique() #has to be str not lst
    print("all unique")
else: print(temp_len)

y = "Churn"
pos_str = "Yes" #1869 
neg_str = "No" #5163

df_f = train_test_split(df_fix, y, pos_str, neg_str)
df_f.to_csv("Telco_final.csv", index=False)

""" Data 2 """
file_name = "Churn_Modelling.csv"
cont_lst = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]

os.chdir(file_loc)
df = pd.read_csv(file_name)
col_lst = list(df.columns)

missing_value_idx(df, col_lst) # No missing values

df_fix = var_type_fix (df, cont_lst)

#ID uniqueness check
unq_var = "CustomerId"
temp_len = len(df_fix[unq_var].unique())
if  temp_len == len(df_fix): # df[unq_var].unique() #has to be str not lst
    print("all unique")
else: print(temp_len)

y = "Exited"
pos_str = "1" # 2037
neg_str = "0" # 7963

df_f = train_test_split(df_fix, y, pos_str, neg_str)
df_f.to_csv("Bank_final.csv", index=False)