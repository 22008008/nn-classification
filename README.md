# Developing a Neural Network Classification Model

### AIM:

To develop a neural network classification model for the given dataset.


### Problem Statement:

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.


### Neural Network Model:

![228911558-37dfca33-87ea-4d0d-bad1-8f693a10786c](https://github.com/22008008/nn-classification/assets/118343520/459fbda3-d385-4e07-9a6d-4d35c911e982)


### DESIGN STEPS:

### STEP 1:
Load the csv file and then use the preprocessing steps to clean the data

### STEP 2:
Split the data to training and testing

### STEP 3:
Train the data and then predict using Tensorflow

### PROGRAM:
```
Developed by:Sri Ranjani Priya .P
register no.:212222220049

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

df = pd.read_csv('/content/customers (1).csv')
df.head()
df.columns
df.dtypes
df.shape
df.isnull().sum()
df_cleaned=df.dropna(axis=0)
df_cleaned.isnull().sum()
df_cleaned.shape
df_cleaned.dtypes
df_cleaned['Gender'].unique()
df_cleaned['Ever_Married'].unique()
df_cleaned['Graduated'].unique()
df_cleaned['Family_Size'].unique()
df_cleaned['Var_1'].unique()
df_cleaned['Spending_Score'].unique()
df_cleaned['Profession'].unique()
df_cleaned['Segmentation'].unique()

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categories_list=[['Male','Female'],
                 ['No','Yes'],
                 ['No','Yes'],
                 ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
                 'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
                 ['Low','Average','High']
                 ]
enc = OrdinalEncoder(categories = categories_list)

cust_1=df_cleaned.copy()
cust_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']]=enc.fit_transform(cust_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

cust_1.dtypes


le = LabelEncoder()
     

cust_1['Segmentation'] = le.fit_transform(cust_1['Segmentation'])
     

cust_1.dtypes
     

cust_1 = cust_1.drop('ID',axis=1)
cust_1 = cust_1.drop('Var_1',axis=1)
     

cust_1.dtypes
     

# Calculate the correlation matrix
corr = cust_1.corr()

# Plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

sns.pairplot(cust_1)

cust_1.describe()
cust_1['Segmentation'].unique()

X=cust_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1 = cust_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray() 
y.shape    
y1[0]
y[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

# To scale the Age column
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)


# Creating the model
ai_brain=Sequential([
    Dense(10,input_shape=(8,)),
    Dense(12,activation='relu'),
    Dense(4,activation='softmax')
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)


ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_
predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
     
# Saving the Model
ai_brain.save('customer_classification_model.h5')

# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,cust_1,df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

ai_brain = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)
     
#Prediction for a single input
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```

### Dataset Information:

![228911165-691c988e-173d-46bb-885b-fcb040db5eb6](https://github.com/22008008/nn-classification/assets/118343520/06275285-b0bc-492d-9d05-c6a0447e0457)

### OUTPUT:

![228910228-057cf2d4-5f06-4c7a-85d1-fbb78889ae8f](https://github.com/22008008/nn-classification/assets/118343520/d6e584e1-b9d6-4c64-8697-4cc47ade8bf8)


### Raining Loss, Validation Loss Vs Iteration Plot:

![228910228-057cf2d4-5f06-4c7a-85d1-fbb78889ae8f](https://github.com/22008008/nn-classification/assets/118343520/32a1440b-1df0-424e-aff4-394ea694206f)

### Classification Report:

![228910620-13e0440c-4d0f-424d-9922-1cbdd0306f5b](https://github.com/22008008/nn-classification/assets/118343520/62625e99-7ec2-4214-9127-d10f1a0d99cf)


### Confusion Matrix:

![228910703-2d756679-4eed-463c-8e54-522ca2da5a2d](https://github.com/22008008/nn-classification/assets/118343520/a6201e49-b758-4d01-afd1-9fc09f234a2e)

### New Sample Data Prediction:

![228910877-a0837d27-3bfc-4d0c-a665-05ddbc515537](https://github.com/22008008/nn-classification/assets/118343520/dfbee373-3b32-48b8-91d1-7ca57d5a6cc2)

### RESULT:

Thus, a Simple Neural Network Classification Model is developed successfully.
