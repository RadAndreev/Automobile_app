# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:17:03 2023

@author: Rado
"""

import pandas as pd
import xgboost as xgb
import pickle 
import streamlit as st

project_path = st.session_state['key']
filename = project_path+'/pickles/data.dat'
data_working = pickle.load(open(filename, 'rb'))

numeric_columns = ['symboling', 'normalized-losses','wheel-base', 'length', 'width', 'height', 
                  'curb-weight','engine-size', 'bore', 'stroke','compression-ratio', 
                  'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']
ordinal_columns = ['num-of-cylinders']
categorical_columns = ['fuel-type' , 'aspiration' , 'num-of-doors' , 'engine-location','make','body-style','drive-wheels','engine-type','fuel-system']


st.title('Automobile price prediction')
st.write('From the various drop-down menus please select inputs')

# Loading model and data types
model_filename = project_path+'/pickles/xgb_model.dat'
xgb_model = pickle.load(open(model_filename, 'rb'))

# read the model from disc
filename = project_path+'/pickles/data_types.dict'
X_data_types = pickle.load(open(filename, 'rb'))

# Selecting model inputs
make_choice = st.selectbox('Select make',data_working['make'].unique())
horsepower_choice = st.selectbox('Select horsepower',data_working['horsepower'].unique())
fueltype_choice = st.selectbox('Select fuel-type',data_working['fuel-type'].unique())
peakrpm_choice = st.selectbox('Select peak-rpm',data_working['peak-rpm'].unique())
citympg_choice = st.selectbox('Select city-mpg',data_working['city-mpg'].unique())
highwaympg_choice = st.selectbox('Select highway-mpg',data_working['highway-mpg'].unique())
numofcylinders_choice = st.selectbox('Select num-of-cylinders',data_working['num-of-cylinders'].unique())
normalizedlosses_choice = st.selectbox('Select normalized-losses',data_working['normalized-losses'].unique()[1:])
wheelbase_choice = st.selectbox('Select wheel-base',data_working['wheel-base'].unique())
length_choice = st.selectbox('Select length',data_working['length'].unique())
width_choice = st.selectbox('Select width',data_working['width'].unique())
heigth_choice = st.selectbox('Select height',data_working['height'].unique())
curbweight_choice = st.selectbox('Select curb-weight',data_working['curb-weight'].unique())
enginesize_choice = st.selectbox('Select engine-size',data_working['engine-size'].unique())
bore_choice = st.selectbox('Select bore',data_working['bore'].unique())
stroke_choice = st.selectbox('Select stroke',data_working['stroke'].unique())
compressionratio_choice = st.selectbox('Select compression-ratio',data_working['compression-ratio'].unique())
aspiration = st.selectbox('Select aspiration',data_working['aspiration'].unique())
numofdoors_choice = st.selectbox('Select num-of-doors',data_working['num-of-doors'].unique())
enginelocation_choice = st.selectbox('Select engine-location',data_working['engine-location'].unique())
bodystyle_choice = st.selectbox('Select body-style',data_working['body-style'].unique())
drivewheels_choice = st.selectbox('Select drive-wheels',data_working['drive-wheels'].unique())
enginetype_choice = st.selectbox('Select engine-type',data_working['engine-type'].unique())
guelsystem_choice = st.selectbox('Select fuel-system',data_working['fuel-system'].unique())
symboling_choice = st.selectbox('Select symboling',data_working['symboling'].unique())

make_choice = 'chevrolet'
prediction_df = pd.DataFrame()
prediction_df.loc[0,'make'] = make_choice
prediction_df.loc[0,'horsepower'] = horsepower_choice
prediction_df.loc[0,'fuel-type'] = fueltype_choice
prediction_df.loc[0,'peak-rpm'] = peakrpm_choice
prediction_df.loc[0,'city-mpg'] = citympg_choice
prediction_df.loc[0,'highway-mpg'] = highwaympg_choice
prediction_df.loc[0,'num-of-cylinders'] = numofcylinders_choice
prediction_df.loc[0,'normalized-losses'] = normalizedlosses_choice
prediction_df.loc[0,'wheel-base'] = wheelbase_choice
prediction_df.loc[0,'length'] = length_choice
prediction_df.loc[0,'width'] = width_choice
prediction_df.loc[0,'height'] = heigth_choice
prediction_df.loc[0,'curb-weight'] = curbweight_choice
prediction_df.loc[0,'engine-size'] = enginesize_choice
prediction_df.loc[0,'bore'] = bore_choice
prediction_df.loc[0,'stroke'] = stroke_choice
prediction_df.loc[0,'compression-ratio'] = compressionratio_choice
prediction_df.loc[0,'aspiration'] = aspiration
prediction_df.loc[0,'num-of-doors'] = numofdoors_choice
prediction_df.loc[0,'engine-location'] = enginelocation_choice
prediction_df.loc[0,'body-style'] = bodystyle_choice
prediction_df.loc[0,'drive-wheels'] = drivewheels_choice
prediction_df.loc[0,'engine-type'] = enginetype_choice
prediction_df.loc[0,'fuel-system'] = guelsystem_choice
prediction_df.loc[0,'symboling'] = symboling_choice

# Inference data creation
feature_names = xgb_model.get_booster().feature_names
dummified_cat_df = pd.get_dummies(prediction_df[categorical_columns])
X_test = pd.concat([prediction_df,dummified_cat_df],axis=1)

for col in feature_names:
    if col not in X_test.columns:
        X_test[col] = 0

X_test = X_test[feature_names] 
X_test = X_test.astype(X_data_types)

y_pred = xgb_model.predict(X_test)
st.write('')
st.title('With the selected values, the predicted price for the automobile is :green['+'{:,}'.format(int(y_pred[0]))+'$]')
