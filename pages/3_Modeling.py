# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:17:03 2023

@author: Rado
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle 
import statsmodels.api as sm
from sklearn.model_selection import cross_validate, GridSearchCV
import streamlit as st
from xgboost import plot_importance
import math

project_path = st.session_state['key']
filename = project_path+'/pickles/data.dat'
data_working = pickle.load(open(filename, 'rb'))


numeric_columns = ['symboling', 'normalized-losses','wheel-base', 'length', 'width', 'height', 
                  'curb-weight','engine-size', 'bore', 'stroke','compression-ratio', 
                  'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']
ordinal_columns = ['num-of-cylinders']
categorical_columns = ['fuel-type' , 'aspiration' , 'num-of-doors' , 'engine-location','make','body-style','drive-wheels','engine-type','fuel-system']
target_column = ['price']



# ---------------- Creating train and test datasets ---------------------------

st.title('Data preparation:')
st.write('Target variable is price. Rows with no price are removed.')
st.write('Special symbol - ? sign - replaced with np.nan')
st.write('One Hot Encoding used for dummification of categorical columns, with 1 column per dummy category removed. This is to escape Dummy Variable Trap.')
st.write('Categorical columns:')
st.write(categorical_columns)
st.write('Ordinal column replaced with numerical value.')
st.write('Ordinal columns:')
st.write(ordinal_columns)
st.write('Missing values for numerical variables imputed with column mean. XGBoost can work with nan values, but in order to train OLS,linear model and obtain cross validation, nan values have to be removed from the dataset.')
st.write('Numerical variables:')
st.write(numeric_columns)


# Dummyfying categorical variables with one hot encoder and creating train-test splits

X = data_working[numeric_columns]
y = data_working[target_column]

dummified_cat_df = pd.get_dummies(data_working[categorical_columns],drop_first=True)
X = pd.concat([X,dummified_cat_df],axis=1)
X=X.fillna(X.mean())

X_data_types = X.dtypes.to_dict()

# Save data types to disk for use in predictions. Uncomment if needed.
#filename = project_path+'/pickles/data_types.dict'
#pickle.dump(X_data_types, open(filename, 'wb'))


X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,test_size = 0.2, random_state=100)

# scoring for cross-validation
scoring = ['neg_mean_squared_error','neg_root_mean_squared_error','neg_mean_absolute_error','r2']

# ---------------- Baseline model ---------------------------------------------
st.title('Modeling steps:')
st.title('1. Train baseline model')
st.write('Train a statsmodels Ordinary Least Squares (OLS) model.')
st.write('Train an sklearn Linear Model with the important features provided the OLS summary - those with p < 0.05 .')
st.write('Record model performance.')

lm_n = sm.OLS(y_train, X_train).fit()
lm_n.summary()
importances_df = pd.DataFrame(data = lm_n.pvalues,columns=['importance']).reset_index()
importances_df.fillna(1,inplace=True)
important_features = list(importances_df[importances_df['importance']<=0.05]['index'])

# instantiate
lm = LinearRegression()
# fit
lm.fit(X_train[important_features], y_train)
# predict 
y_pred = lm.predict(X_test[important_features])


# metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mse) #rmse = mean_squared_error(y_test, y_pred,squared=True)
r2 = r2_score(y_true=y_test, y_pred=y_pred) 
 
st.write("Mean Squared Error (MSE):", np.round(mse,0)) 
st.write("Mean Absolute Error (MAE):", np.round(mae,0)) 
st.write("Root Mean Squared Error (RMSE):", np.round(rmse,0)) 
st.write("R-squared (R2) Score:", np.round(r2,4)) 
st.write('If only looking and error metrics, model has good error metrics ')


st.write('Perform cross-validation to check metrics behaviour in folds. The difference in metrics shows overfitting.')

# cross-validation
scores = cross_validate(lm, X[important_features], y, cv=5,scoring=scoring)
scores_df = pd.DataFrame(data=scores)
scores_df.rename(columns={'test_neg_mean_squared_error':'test_mse',
                          'test_neg_root_mean_squared_error':'test_rmse',
                          'test_neg_mean_absolute_error':'test_mae'},inplace=True)
scores_df = scores_df[['test_mse', 'test_rmse', 'test_mae','test_r2']].round(2)

scores_df[['test_mse', 'test_rmse', 'test_mae']]= scores_df[['test_mse', 'test_rmse', 'test_mae']] * -1
scores_df.index.name = 'fold_number'
st.write(scores_df)

# ---------------- Plotting ---------------------------------------------------

residuals_df = y_test.copy()
residuals_df['y_pred'] = y_pred 
residuals_df['residuals'] = residuals_df['price'] - residuals_df['y_pred']

fig_1, ax_1 = plt.subplots()
sns.regplot(x = residuals_df['y_pred'], y = residuals_df['price'], data = None, scatter = True, color = 'blue')
plt.title('Actual vs Predicted plot',pad=10)
st.pyplot(fig_1)

fig_2, ax_2 = plt.subplots()
sns.regplot(x = residuals_df['y_pred'], y = residuals_df['residuals'], data = None, scatter = True, color = 'red')
plt.title('Residual plot',pad=10)
st.pyplot(fig_2)
st.write('Baseline optimized linear model is not a bad model, it has predictive value. Both plots are healthy.')

# ------------- Xgboost data and model ----------------------------------------
# ------------- Hyperparameter tuninig ----------------------------------------
# ------------- Best model already selected -----------------------------------

st.title('2. Train xgboost model.')
st.write('Perform Gridsearch for hyperparameter optimization')

# # Define the hyperparameter grid
# param_grid = {
#     'n_estimators':[100,200,500,1000],
#     'max_depth': [3,6,9],
#     'learning_rate': [0.1, 0.03],
#     'subsample': [0.5,0.7],
#     'colsample_bytree' : [0.5,0.8]
# }

# # Create the XGBoost model object
# xgb_model = xgb.XGBRegressor()

# # Create the GridSearchCV object
# grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error')


# # Fit the GridSearchCV object to the training data
# grid_search.fit(X_train, y_train)

# # Print the best set of hyperparameters and the corresponding score
# print("Best set of hyperparameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)


st.write('Train XGBRegressor')
st.write('Record model performance')
st.write('Pickle model')

# instantiate
xgb_model = xgb.XGBRegressor(learning_rate =0.2,n_estimators=80, max_depth=8, subsample=0.7, colsample_bytree=0.8)   

# fit
xgb_model.fit(X_train, y_train)

# predict
y_pred = xgb_model.predict(X_test)
predictions = [round(value) for value in y_pred]

# metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mse) #rmse = mean_squared_error(y_test, y_pred,squared=True)
r2 = r2_score(y_true=y_test, y_pred=y_pred) 
 
st.write("Mean Squared Error (MSE):", np.round(mse,0))
st.write("Mean Absolute Error (MAE):", np.round(mae,0)) 
st.write("Root Mean Squared Error (RMSE):", np.round(rmse,0))
st.write("R-squared (R2) Score:", np.round(r2,4)) 
st.write('Perform cross-validation to check metrics behaviour in folds. Error spread in different folds is closer than in the linear model, so the xgboost shows to be the healthier model.')

# cross-validation
scores = cross_validate(xgb_model, X, y, cv=5,scoring=scoring)
scores_df = pd.DataFrame(data=scores)
scores_df.rename(columns={'test_neg_mean_squared_error':'test_mse',
                          'test_neg_root_mean_squared_error':'test_rmse',
                          'test_neg_mean_absolute_error':'test_mae'},inplace=True)
scores_df = scores_df[['test_mse', 'test_rmse', 'test_mae','test_r2']].round(2)

scores_df[['test_mse', 'test_rmse', 'test_mae']]= scores_df[['test_mse', 'test_rmse', 'test_mae']] * -1
scores_df.index.name = 'fold_number'
st.write(scores_df)

# ---------------------------- Plotting ---------------------------------------
residuals_df = y_test.copy()
residuals_df['y_pred'] = y_pred 
residuals_df['residuals'] = residuals_df['price'] - residuals_df['y_pred']

fig_1, ax_1 = plt.subplots()
sns.regplot(x = residuals_df['y_pred'], y = residuals_df['price'], data = None, scatter = True, color = 'blue')
plt.title('Actual vs Predicted plot',pad=10)
st.pyplot(fig_1)
st.write('Actual vs Predicted plot shows that the model is good as most points are within the diagonal. Outliers are few.')


fig_2, ax_2 = plt.subplots()
sns.regplot(x = residuals_df['y_pred'], y = residuals_df['residuals'], data = None, scatter = True, color = 'red')
plt.title('Residual plot',pad=10)
st.pyplot(fig_2)
st.write('Residual plot is healthy, most prediction errors are around 0. There are outliers that can be looked into and removed from the set if they truly are outliers.')


st.pyplot(plot_importance(xgb_model,max_num_features=15).figure)

st.write('Showing 15 most important features. Although one is with most weight, all the other features are not with negligible importance.')
st.write('The linear model carries the issue of multicollinearity as present in the data and correlation matrix.')
st.write('Xgboost is selected for the objectively better performance, immunity to multicollinearity and decreased overfitting as shown in the cross-validation metrics.')

# Save the model to disk. Uncomment if needed.
#filename = project_path+'/pickles/xgb_model.dat'
#pickle.dump(xgb_model, open(filename, 'wb'))



