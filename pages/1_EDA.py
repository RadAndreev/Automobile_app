# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:17:03 2023

@author: Rado
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 
import streamlit as st

project_path = st.session_state['key']
data = pd.read_csv(project_path+'/data/Automobile_data.csv')

    
data_working = data[data['price']!='?']
data_working = data_working.replace('?',np.nan)
data_working.reset_index(inplace=True,drop=True)


data_working['num-of-cylinders'].unique()
cylinder_dict = {'two':2, 'three':3,'four':4,'five':5, 'six':6, 'eight':8,'twelve':12} 
data_working['num-of-cylinders'] = data_working['num-of-cylinders'].replace(cylinder_dict)

data_working['normalized-losses'] = pd.to_numeric(data_working['normalized-losses'] )
data_working['bore'] = pd.to_numeric(data_working['bore'] )
data_working['stroke'] = pd.to_numeric(data_working['stroke'] )
data_working['horsepower'] = pd.to_numeric(data_working['horsepower'] )
data_working['peak-rpm'] = pd.to_numeric(data_working['peak-rpm'] )
data_working['price'] = pd.to_numeric(data_working['price'] )
data_working['num-of-cylinders'] = pd.to_numeric(data_working['num-of-cylinders'] )

# Save data to disk for use in other pages. Uncomment if needed.
#filename = project_path+'/pickles/data.dat'
#pickle.dump(data_working, open(filename, 'wb'))
# pickle data for use in other parts of the app

# ----------------- EDA -------------------------------------------------------

st.title('Exploratory Data Analysis')
st.write('')
st.write('')

# ----------------- Manufacturers ---------------------------------------------

fig_1, ax_1 = plt.subplots()
working_df=data_working[['make','price']].copy()
working_df.sort_values(by='price', inplace=True)
sns.boxplot(x = 'make', y = 'price', data=working_df)
plt.xticks(rotation=70)
plt.tight_layout()
plt.title('Price distribution of different models/makes',pad=10)
st.pyplot(fig_1)

st.write('The bivariate plot of manufacturer and price gives a good graphical image of the data and shows the varians in the targer variable.') 
st.write('The general comparison of price among the different manufacturers is one of the most often talked about colloquial measure when it comes to buying a new car.')
st.write('')
st.write('')


# ----------------- Scatter plots ---------------------------------------------

fig_2, axs_2 = plt.subplots(4, 4,  figsize=(14, 7),layout='constrained')
sns.scatterplot(x=data_working['wheel-base'], y=data_working['price'], ax=axs_2[0, 0])
sns.scatterplot(x=data_working['length'], y=data_working['price'], ax=axs_2[0, 1])
sns.scatterplot(x=data_working['width'], y=data_working['price'], ax=axs_2[0, 2])
sns.scatterplot(x=data_working['curb-weight'], y=data_working['price'], ax=axs_2[0, 3])
sns.scatterplot(x=data_working['engine-size'], y=data_working['price'], ax=axs_2[1, 0])
sns.scatterplot(x=data_working['bore'], y=data_working['price'], ax=axs_2[1, 1])
sns.scatterplot(x=data_working['stroke'], y=data_working['price'], ax=axs_2[1, 2])
sns.scatterplot(x=data_working['compression-ratio'], y=data_working['price'], ax=axs_2[1, 3])
sns.scatterplot(x=data_working['horsepower'], y=data_working['price'], ax=axs_2[2, 0])
sns.scatterplot(x=data_working['peak-rpm'], y=data_working['price'], ax=axs_2[2, 1])
sns.scatterplot(x=data_working['city-mpg'], y=data_working['price'], ax=axs_2[2, 2])
sns.scatterplot(x=data_working['highway-mpg'], y=data_working['price'], ax=axs_2[2, 3])
sns.scatterplot(x=data_working['num-of-cylinders'], y=data_working['price'], ax=axs_2[3, 0])
sns.scatterplot(x=data_working['symboling'], y=data_working['price'], ax=axs_2[3, 1])
sns.scatterplot(x=data_working['normalized-losses'], y=data_working['price'], ax=axs_2[3, 2])
sns.scatterplot(x=data_working['height'], y=data_working['price'], ax=axs_2[3, 3])
fig_2.suptitle('Scatter plots of numerical features with price',fontsize=20,y=1.05)

st.pyplot(fig_2)

st.write('Price is positively correlated with engine-size,curb-weight, horsepower, and the car sizes width and length.')
st.write('There is negative correlation between the target variable price and city-mpg and highway-mpg.')
st.write('This plot shows is there are patterns in the data which would allow for modeling the price.')
st.write('')
st.write('')

# ----------------- Correlations ----------------------------------------------

corr1 = data_working.corr()
price_corr = pd.DataFrame(data=corr1.loc['price'].reset_index()).sort_values(by=['price'],ascending=False)
price_corr_2 = price_corr[price_corr['price'].abs()>0.5]
main_correlation_factors = list(price_corr_2['index'])
main_correlation_factors.remove('price')
main_correlation_factors.append('price')

corr2 = data_working[main_correlation_factors].corr()
fig_3, ax_3 = plt.subplots()
sns.heatmap(corr2,annot= True,cmap = 'coolwarm')
plt.title('Correlation matrix with price, corr > 0.5',pad=10)
st.pyplot(fig_3)

st.write('Matrix of the variables most correlated with the target variable (correlation > 0.5)')
st.write('As with the scatter plot, dependencies between variables can be derived from the correlation matrix.')
st.write('The main difference from the scatter plot is that we can observe multicollinearity.')
st.write('Multicollinearity will lead to issues in linear models.Techniques like PCA can mitigathe the issue, but will also decrease model explainability.')
st.write('As early as this stage, selecting a tree algorithm would be optimal, as tree and boosted tree models are immune to multicollinearity by nature.')
st.write('')
st.write('')
