# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:17:03 2023

@author: Rado
"""

import streamlit as st

st.title('Data Scientist Technical Assessment')
st.write('')
st.write('')
st.write('1. Using the kaggle automobile dataset, perform the following tasks:')
st.write('2. Exploratory Data Analysis (EDA)')
st.write('3. Multiple Correspondence Analysis (MCA)')
st.write('4. Build model for price prediction')
st.write('5. Predict automobile price')
st.write('6. Make deployment plan')


st.write('')
st.write('')


st.write('Project developer: [Radoslav Andreev](radoslav.andreev@gmail.com)')


# ----- define project directory

project_path = 'D:/Documents/Applications/BlackPeak/Automobile_app'

if 'key' not in st.session_state:
    st.session_state['key'] = project_path 