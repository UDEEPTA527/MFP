#!/usr/bin/env python
# coding: utf-8

# # mfp final

# In[204]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[3]:


data=pd.read_csv("ai4i2020.csv")


# In[4]:


df=data.copy()
df.head(10)


# # MERGE ALL TYPES OF MACHINE FAILURE INTO ONE COLUMN

# In[6]:


df = pd.DataFrame(data)


def determine_failure(row):
    for failure_type in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        if row[failure_type] == 1:
            return failure_type  
    return "No Failure" 

df["Type of Machine Failure"] = df.apply(determine_failure, axis=1)


df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])




# In[7]:


df.head(74)


# In[8]:


df["Machine failure"].value_counts(normalize=True)


# In[9]:


df["Type of Machine Failure"].value_counts(normalize=True)


# In[10]:


df.info()


# In[11]:


df["Type"].value_counts(normalize=True)


# # AS ID COLUMN HAS NO IMPACT IN THE PREDICTION , SO WE DROP 2 COLUMNS

# In[13]:


df1=df.drop(["UDI","Product ID"],axis=1)


# # WE NEED TO SEE IF THERE IS ANY CONTRADICTORY BETWEEN THE TWO TARGET VARIBLE

# In[15]:


df_failure=df1[df1["Machine failure"]==1]


# In[16]:


df_failure.head()


# In[17]:


df_failure["Type of Machine Failure"].value_counts()


# In[18]:


df_failure[df_failure["Type of Machine Failure"]=="No Failure"]


# # DROP THESE OBSERVATION BECAUSE WE ARE NOT SURE OF THE REASON OF FAILURE

# In[20]:


target_drop=df_failure[df_failure["Type of Machine Failure"]=="No Failure"].index


# In[21]:


target_drop


# In[22]:


df1=df1.drop(target_drop,axis=0)


# In[23]:


df.shape


# In[24]:


df1.shape


# # WE NEED TO SEE IF THERE IS ANY CONTRADICTORY BETWEEN THE TWO TARGET VARIBLE

# In[26]:


df_no_failure=df1[df1["Machine failure"]==0]


# In[27]:


df_no_failure.head()


# In[28]:


df_no_failure["Type of Machine Failure"].value_counts()


# In[29]:


df_no_failure[df_no_failure["Type of Machine Failure"]=="RNF"]


# # DROP THESE OBSERVATION BECAUSE WE ARE NOT SURE REASON OF FAILURE

# In[31]:


target_no_failure_drop=df_no_failure[df_no_failure["Type of Machine Failure"]=="RNF"].index


# In[32]:


target_no_failure_drop


# In[33]:


df2=df1.drop(target_no_failure_drop,axis=0)


# In[34]:


df2.shape


# # EDA TO FIND THE REASON OF FAILURE

# In[36]:


sns.pairplot(df2,hue="Machine failure")


# # EDA FOR TYPES OF MACHINE FAILURE

# In[38]:


sns.pairplot(df2,hue="Type of Machine Failure")


# # DATA SHEET

# In[40]:


df2.head()


# # EDA OF ROTATIONAL SPEED AND TORQUE

# In[42]:


df100=df2[["Rotational speed [rpm]","Torque [Nm]"]]


# In[43]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df2[df2["Machine failure"]==1],x="Rotational speed [rpm]",y="Torque [Nm]",hue="Type of Machine Failure")


# # CORRELATION BETWEEN COLUMNS

# In[45]:


df2.head()


# In[46]:


# Select only numeric columns for correlation
numeric_df = df2.select_dtypes(include=['number'])

# Calculate correlation and plot heatmap
sns.heatmap(numeric_df.corr(), annot=True)


# # percentage of failure based on quality of components L, M, H

# In[48]:


Low_quality=df2[df2["Type"]=="L"]
Low_quality


# In[49]:


Low_quality["Machine failure"].value_counts(normalize=True)*100


# In[50]:


MEDIUM_quality=df2[df2["Type"]=="M"]
MEDIUM_quality["Machine failure"].value_counts(normalize=True)*100


# In[51]:


HIGH_quality=df2[df2["Type"]=="H"]
HIGH_quality["Machine failure"].value_counts(normalize=True)*100


# # ALTHOUGH THE LOW PRICE COMPONENT HAS FAILURE AMOUNT 3.86 WITH RESPECT TO MEDIUM 2.64 AND HIHG PRICE 2.00 , BUT THE DIFFERENCE IS NOT VERY HIGH

# # FINDING THE OUTLIER

# In[54]:


sns.boxplot(data=df2,x="Torque [Nm]")


# In[55]:


sns.boxplot(data=df2,x="Rotational speed [rpm]")


# # Label Encoding

# In[57]:


df2["Type"]=df2["Type"].map({"L":0,"M":1,"H":2})


# In[58]:


df2["Type"].value_counts()


# # AFTER CLEANING THE DATA FILE

# In[60]:


# Save df2 as a CSV file
df2.to_csv("cleaned_data.csv", index=False)


# In[61]:


df2.describe()


# # DROP THE PREDICTION COLUMN

# In[63]:


from sklearn.feature_selection import chi2
# Drop the target columns from X
X = df2.drop(columns=["Machine failure", "Type of Machine Failure"], axis=1)

# Assign target columns to y
y = df2["Machine failure"]



# In[64]:


df2.head()


# # TRAIN TEST SPLIT

# In[66]:


# import library
from sklearn.model_selection import train_test_split
# split the data into train, test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Model Training

# In[68]:


from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def classify(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
    model.fit(X_train, y_train)   
    pred_test = model.predict(X_test)
    print("Testing Accuracy_Score is", accuracy_score(pred_test, y_test) * 100)
    
    # K-Fold Stratified Cross Validation:
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    score = cross_val_score(model, X, y, cv=stratified_kfold)
    print("Accuracy using K-Fold Stratified Cross Validation is,", np.mean(score) * 100)


# # Logistic Regression

# In[70]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# # DecisionTree

# In[72]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)


# # Random Forest

# In[74]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model,X,y)


# # AdaBoost

# In[76]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
classify(model,X,y)


# # KNeighbors

# In[78]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=20)
classify(model,X,y)


# # Naive Bayes

# In[80]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
classify(model,X,y)


# # Support Vector Machine

# In[82]:


from sklearn import svm
model = svm.SVC(kernel='linear')
classify(model,X,y)


# # Confusion Matrix

# In[84]:


from sklearn.metrics import confusion_matrix

def cf_matrix(model, X, y):
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    cm = confusion_matrix(pred_test, y_test)
    sns.heatmap(cm, annot = True, fmt = 'g')
    # Classification Report
    print(classification_report(pred_test,y_test))


# In[85]:


# Logistic Regression
model = LogisticRegression()
cf_matrix(model, X, y)


# In[86]:


# Decision Tree Classifier
model = DecisionTreeClassifier()
cf_matrix(model, X, y)


# In[87]:


# Random Forest Classifier
model = RandomForestClassifier()
cf_matrix(model, X, y)


# In[88]:


# Ada Boost Classifier
model = AdaBoostClassifier()
cf_matrix(model, X, y)


# In[89]:


# KNeighbors Classifier
model = KNeighborsClassifier(n_neighbors=20)
cf_matrix(model, X, y)


# In[90]:


# Naive Bayes Classifier
model = GaussianNB()
cf_matrix(model, X, y)


# In[91]:


# Support Vector Classifier
model = svm.SVC(kernel='linear')
cf_matrix(model,X,y)


# # Conclusion
# 
# Model Accuracies are:
# 
# 1. Logistic Regression              97.06225
# 2. Desicion Tree Classifier         98.114701
# 3. Random Forest Classifier       .    4278
# 5. Ada Boost Classifier            97.5.417
# 6. KNeighbor Classifier         .   968.710
# 7. Naive Bayes f      i      er.    780.417
# 8. Support Vector                     80.417

# # Best Model Implementation

# In[94]:


#lg_rg = LogisticRegression(solver= 'liblinear', penalty = 'l1', max_iter = 300, C = 1)
#lg_rg.fit(X,y)
from sklearn.ensemble import RandomForestClassifier

# Define hyperparameters as a dictionary
hyperparameters = {
    'n_estimators': 900,
    'min_samples_split': 2,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'max_depth': 20
}

# Create the RandomForestClassifier instance with the specified hyperparameters
rf_cl = RandomForestClassifier(**hyperparameters)
rf_cl.fit(X, y)


# In[95]:


import joblib
# Save the model
joblib.dump(rf_cl,"Machine failure")

# Load the model
model = joblib.load("Machine Failure")


# In[96]:


df3 = pd.DataFrame({'Type':1, 'Air temperature [K]':298.1, 'Process temperature [K]':308.6, 'Rotational speed [rpm]':1551, 'Torque [Nm]':42.8, 'Tool wear [min]':0},index = [0])


# In[97]:


df3


# In[98]:


result = model.predict(df3)
if result == 1:
    print("Model Prediction: Machine Failed")
else:
    print("Model Prediction: Machine not Failed")


# In[99]:


df3 = pd.DataFrame({'Type':0, 'Air temperature [K]':298.9, 'Process temperature [K]':309.0, 'Rotational speed [rpm]':1410, 'Torque [Nm]':65.7, 'Tool wear [min]':191},index = [0])


# In[100]:


result = model.predict(df3)
if result == 1:
    print("Model Prediction: Machine Failed")
else:
    print("Model Prediction: Machine not Failed")


# In[188]:


import joblib

# Save the model to a .pkl file
joblib.dump(rf_cl, "Machine_failure.pkl")


# In[190]:


# Load the model from the .pkl file
model = joblib.load("Machine_failure.pkl")


# In[101]:


# Filter rows where machine failure occurred for the second model
failure_data = df2[df2['Machine failure'] == 1]
X_failure_type = failure_data[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y_failure_type = failure_data['Type of Machine Failure']

X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X_failure_type, y_failure_type, test_size=0.2, random_state=42)

# Train Model 2: Type of Failure
failure_type_model = RandomForestClassifier(random_state=42)
failure_type_model.fit(X_train_type, y_train_type)

# Evaluate Model 2
y_pred_type = failure_type_model.predict(X_test_type)
print("\nType of Failure Prediction Accuracy:", accuracy_score(y_test_type, y_pred_type))

# Save the failure type model
joblib.dump(failure_type_model, 'failure_type_model.pkl')


# In[102]:


def predict_failure_and_type(data):
    """
    Predict machine failure and type of failure.
    
    Parameters:
        data (pd.DataFrame): Input data containing required features.

    Returns:
        pd.DataFrame: Predictions for machine failure and type of failure.
    """
    # Load models
    failure_model = joblib.load('Machine Failure')
    failure_type_model = joblib.load('failure_type_model.pkl')
    
    # Predict machine failure
    X = data[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    failure_predictions = failure_model.predict(X)
    
    # Initialize the result
    results = data.copy()
    results['predicted_machine_failure'] = failure_predictions
    results['predicted_failure_type'] = None
    
    # Predict type of failure where failure is predicted
    failure_indices = results[results['predicted_machine_failure'] == 1].index
    if not failure_indices.empty:
        failure_data = X.loc[failure_indices]
        type_predictions = failure_type_model.predict(failure_data)
        results.loc[failure_indices, 'predicted_failure_type'] = type_predictions

    return results


# In[103]:


# Example usage with test data
test_data = pd.read_csv('cleaned_data.csv')
predictions = predict_failure_and_type(test_data)
print(predictions[['predicted_machine_failure', 'predicted_failure_type']])


# In[104]:


df3 = pd.DataFrame({'Type':0, 'Air temperature [K]':298.9, 'Process temperature [K]':309.0, 'Rotational speed [rpm]':1410, 'Torque [Nm]':65.7, 'Tool wear [min]':191},index = [0])
predictions = predict_failure_and_type(df3)
print(predictions[['predicted_machine_failure', 'predicted_failure_type']])


# In[192]:


import pickle


# In[194]:


# Assuming rf_cl and failure_type_model are your trained models
with open('Machine_failure.pkl', 'wb') as file:
    pickle.dump(rf_cl, file)

with open('failure_type_model.pkl', 'wb') as file:
    pickle.dump(failure_type_model, file)


# In[196]:


# Assuming rf_cl and failure_type_model are your trained models
with open('Machine_failure.pkl', 'wb') as file:
    pickle.dump(rf_cl, file)

with open('failure_type_model.pkl', 'wb') as file:
    pickle.dump(failure_type_model, file)


# In[198]:


# Load the Machine_failure model
with open('Machine_failure.pkl', 'rb') as file:
    loaded_rf_cl = pickle.load(file)

# Load the failure_type_model
with open('failure_type_model.pkl', 'rb') as file:
    loaded_failure_type_model = pickle.load(file)


# In[200]:


import pickle

# Load the .pkl file
with open('machine_failure.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

# Save it to a .sav file
with open('machine_failure.sav', 'wb') as sav_file:
    pickle.dump(model, sav_file)

print("Model successfully converted to .sav format.")


# In[202]:


import pickle

# Load the .pkl file
with open('failure_type_model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

# Save it to a .sav file
with open('failure_type_model.sav', 'wb') as sav_file:
    pickle.dump(model, sav_file)

print("Model successfully converted to .sav format.")


# In[ ]:




