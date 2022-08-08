# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 12:25:11 2022

@author: chelsea.wang
"""

"""   
This script is aimed to explore Neural Network model for both datasets
MUST complete the following requirements before running this script
1. Run Script for Step1 which contains all required modules and functions
2. Download the Cleaned data files and save them in the same folder with all scripts
3. Navigate to the folder with Cleaned data files
"""

""" NN Prep FOR Telcom"""

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
X_train = df_ml[df_ml["set_type"] == "train"].drop(y_col + other_col, axis=1) 
Y_train =df_ml[df_ml["set_type"] == "train"][y_col[0]]
Y_train_ref = Y_train.apply(lambda x: 1 if x== "Yes" else 0)

x_test = df_ml[df_ml["set_type"] == "test"].drop(y_col + other_col, axis=1) 
y_test =df_ml[df_ml["set_type"] == "test"][y_col[0]]
y_test_ref = y_test.apply(lambda x: 1 if x== "Yes" else 0)


""" NN training Teclcom """
#NOTE: different arcitechures (neuron size, # of hidden layer, lr) were explored for both data sets 
# Please refer to analysis report for details.
# Only selected arcitechures are shown below, please change accordinly to test specific model's performance

tf.random.set_seed(123)
#https://towardsdatascience.com/how-to-train-a-classification-model-with-tensorflow-in-10-minutes-fd2b7cfba86 
model = tf.keras.Sequential([
    tf.keras.layers.Input(name="input", shape=(40,)), #input layer
    tf.keras.layers.Dense(25, activation='sigmoid'), #hidden layer
    #tf.keras.layers.Dense(10, activation='relu'), #hidden layer2
    tf.keras.layers.Dense(1, activation='sigmoid') #output layer
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.0005, epsilon = 0.005),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(X_train, Y_train_ref, epochs=200, batch_size=32, shuffle=True, validation_split=0.2)

title = "Neural Network (1 hidden layer/sigmoid, 25 neurons, lr=0.0005)"
NN_learning_curve(history, title, epoch=200, y_max = 1)

""" NN Final Teclcom 25 N, LR=0.0005, signoid"""

tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Input(name="input", shape=(40,)), #input layer
    tf.keras.layers.Dense(25, activation='sigmoid'), #hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid') #output layer
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon = 0.005),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

history = model.fit(X_train, Y_train_ref, epochs=200, batch_size=32, shuffle=True, validation_split=0.2)
#TEST
predictions = model.predict(x_test)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
confusion_matrix(y_test_ref, prediction_classes)

timetaken = timecallback()
model.fit(X_train, Y_train_ref, epochs=200, batch_size=32, shuffle=True, validation_split=0.2, callbacks = [timetaken])
#~50 s for Telcom

#Train
predictions = model.predict(X_train)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
confusion_matrix(Y_train_ref, prediction_classes)

""" NN Prep FOR Bank"""

df_f = pd.read_csv("Bank_final.csv")
y_col= ["Exited"]
other_col = ["set_type"]
num_cols = ["CreditScore", 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
dummy_cols= ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

# retrieve just the numeric input values
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
X_train = df_ml[df_ml["set_type"] == "train"].drop(y_col + other_col, axis=1) 
Y_train =df_ml[df_ml["set_type"] == "train"][y_col[0]]

x_test = df_ml[df_ml["set_type"] == "test"].drop(y_col + other_col, axis=1) 
y_test =df_ml[df_ml["set_type"] == "test"][y_col[0]]


""" NN training Bank """
#NOTE: different arcitechures (neuron size, # of hidden layer, lr) were explored for both data sets 
# Please refer to analysis report for details.
# Only selected arcitechures are shown below, please change accordinly to test specific model's performance

tf.random.set_seed(123)
#https://towardsdatascience.com/how-to-train-a-classification-model-with-tensorflow-in-10-minutes-fd2b7cfba86 
model = tf.keras.Sequential([
    tf.keras.layers.Input(name="input", shape=(13,)), #input layer
    tf.keras.layers.Dense(20, activation='relu'), #hidden layer
    #tf.keras.layers.Dense(10, activation='relu'), #hidden layer2
    tf.keras.layers.Dense(1, activation='sigmoid') #output layer
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005, epsilon = 0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
timetaken2 = timecallback()
history = model.fit(X_train, Y_train, epochs=200, batch_size=32, shuffle=True, validation_split=0.2, callbacks = [timetaken2])
# ~60s for bank

title = "Neural Network (1 hidden layer/sigmoid), 20 neurons, lr=0.0005)"
NN_learning_curve(history, title, epoch=200, y_max = 1)

""" NN Final Bank relu, 20NN, LR=0.005"""
history = model.fit(X_train, Y_train, epochs=200, batch_size=32, shuffle=True, validation_split=0.2)
#TEST
predictions = model.predict(x_test)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
confusion_matrix(y_test, prediction_classes)

#Train
predictions = model.predict(X_train)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
confusion_matrix(Y_train, prediction_classes)



