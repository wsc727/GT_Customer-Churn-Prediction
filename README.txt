CS7641 Assignment 1 - Shangci Wang (swang879)

Description: This work aims to develop and compare prediction model for two customer churn datasets (Telcom and Bank) by using 5 supervised machine learning algorithms which are KNN, SVM, Decision Tree (DT), AdaBoost Decision Tree (ABDT), and Neural Network (NN). 

Configurations: All the analyses are performed on a PC (windows 10, x64) by using Python (3.7.4) with Spyder IDE (4.1.5). The following modules are required: pandas (1.3.0), numpy(1.19.2), matplotlib(3.3.3), scikit-learn (0.23.2), tensorflow (2.7.0), graphviz (0.16).

Instructions for applications 
All the scripts are in this folder: code. Please download and save them in the same folder with all the data files.

Step 1. Install Python and all the required modules. Run the script named “swang879_Step1_module preparation.py”. This script includes all the modules and functions required for later steps. It is required to run this script before testing other scripts. The raw datasets are available in the following link:
Telcom: WA_Fn-UseC_-Telco-Customer-Churn.csv
Bank: Bank_Churn_Modelling.csv
Note: MUST run script 1 before running any other scripts. MUST save all data files and scripts in the same folder. 

Step 2. This step is aimed to clean raw datasets. User can skip this step as needed by directly download cleaned dataset as csv file from data folder. Open script “swang879_Step2_data cleaning.py”, change file directory accordingly, run this script to clean raw data sets. The cleaned csv files will be exported to the same folder, which are also available in the data folder.

Step 3. This step is to explore KNN model for both datasets. Open script “swang879_Step3_KNN.py”, Navigate to the folder with datafile. Follow the comment in the script to conduct initial model exploration, plot validation and learning curves, generate final KNN model, print its performance. Best parameters returned from initial screening and used in final model creation are in the comments.  
Step 4. This step is to explore SVM model for both datasets. To speed up training, only 20% training data was randomly selected for hyperparameter tuning and validation curve. All the training set was applied for learning curve and final model training. Open script “swang879_Step4_SVM.py”, Navigate to the folder with datafile. Follow the comment in the script to conduct initial model exploration, plot validation and learning curves, generate final SVM model, print its performance. Best parameters returned from initial screening and used in final model creation are in the comments.  
Step 5. This step is to explore DT model for both datasets. Open script “swang879_Step5_DecisionTree.py”, Navigate to the folder with datafile. Follow the comment in the script to conduct initial model exploration, plot validation and learning curves, generate final DT model, print its performance. Best parameters returned from initial screening and used in final model creation are in the comments.  
Step 6. This step is to explore ABDT model for both datasets. Open script “swang879_Step6_AdaBoostDecisionTree.py”, Navigate to the folder with datafile. Follow the comment in the script to conduct initial model exploration, plot validation and learning curves, generate final ABDT model, print its performance. Best parameters returned from initial screening and used in final model creation are in the comments.  
Step 7. This step is to explore NN model for both datasets. Open script “swang879_Step7_NeuralNetwork.py”, Navigate to the folder with datafile. Follow the comment in the script to conduct initial model exploration, plot validation and learning curves, generate final NN model, print its performance. Best parameters returned from initial screening and used in final model creation are in the comments.  

Conclusion: NN is considered to be the best model for Telcom Customer Churn prediction because it has highest test accuracy (80.9%), second highest test precision (66.8%) and recall (56.3%).  AdaBoost decision tree model is considered to be the best option for Bank Customer Churn prediction. It has highest test accuracy (86.5%) among 5 models with test precision noticeably higher than the remaining 2 candidates (75.5% vs. ~70% or less). Its recall (49.8%) is only ~1-2% less than DT (52.1%) and NN (51.3%). For both datasets, fitting time for 5 algorithms rank from high to low in the order of NN >>> SVM ~ ADBT >>> DT ~ KNN. 
