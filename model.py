# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:00:01 2018

@author: Maddy
"""

# Import dependencies
import pandas as pd
import numpy as np

# Load the dataset in a dataframe object and include only four features as mentioned
df = pd.read_csv("claim_API.csv")
include = ['AccidentArea','Sex','MaritalStatus', 'Age', 'Fault', 'VehicleCategory', 'Deductible', 'AgentType', 
           'AddressChange_Claim', 'BasePolicy', 'FraudFound_P'] # Only four features
df_ = df[include]

categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)
          
          
df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

from sklearn.linear_model import LogisticRegression
dependent_variable = 'FraudFound_P'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)


from sklearn.externals import joblib
joblib.dump(lr, 'model.pkl')

lr = joblib.load('model.pkl')

model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')