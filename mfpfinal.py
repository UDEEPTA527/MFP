#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy  as np
import streamlit as st
import warnings

warnings.filterwarnings("ignore")


# In[13]:


df2 = pd.read_csv('cleaned_data.csv')
df2.describe()

X = df2.drop(columns=["Machine failure", "Type of Machine Failure"], axis=1)
y = df2["Machine failure"]

def classify(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
    model.fit(X_train, y_train)   
    pred_test = model.predict(X_test)
    print("Testing Accuracy_Score is", accuracy_score(pred_test, y_test) * 100)
    
    # K-Fold Stratified Cross Validation:
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    score = cross_val_score(model, X, y, cv=stratified_kfold)
    print("Accuracy using K-Fold Stratified Cross Validation is,", np.mean(score) * 100)


# In[15]:


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


# In[19]:


import joblib


# In[21]:


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


# In[23]:


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


# In[29]:


# Example usage with test data
test_data = pd.read_csv('cleaned_data.csv')
predictions = predict_failure_and_type(df2)
print(predictions[['predicted_machine_failure', 'predicted_failure_type']])


# In[31]:


df3 = pd.DataFrame({'Type':0, 'Air temperature [K]':298.9, 'Process temperature [K]':309.0, 'Rotational speed [rpm]':1410, 'Torque [Nm]':65.7, 'Tool wear [min]':191},index = [0])
predictions = predict_failure_and_type(df3)
print(predictions[['predicted_machine_failure', 'predicted_failure_type']])


# In[33]:


import pickle
filename = 'trained_model.sav'
with open(filename, 'wb') as file:
    pickle.dump("failure_type_model.pkl", file)

# Load the model from the .sav file
with open('trained_model.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model for predictions
# Example: predictions = loaded_model.predict(X_test)


# In[ ]:




