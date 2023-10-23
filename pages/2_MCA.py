# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:17:03 2023

@author: Rado
"""

import prince
import pickle 
import streamlit as st


project_path = st.session_state['key']

# read the data from pickle
filename = project_path+'/pickles/data.dat'
data_working = pickle.load(open(filename, 'rb'))

categorical_columns = ['fuel-type' , 'aspiration' , 'num-of-doors' , 'engine-location','make','body-style','drive-wheels','engine-type','fuel-system']

st. title('Multiple Correspondence Analysis')
container = st.container()
all = st.checkbox("Select all")
 
if all:
    selected_options = container.multiselect("Select one or more options:",
         categorical_columns,categorical_columns)
else:
    selected_options =  container.multiselect("Select one or more options:",
        categorical_columns)

st.write(selected_options)

try:
    mca_data = data_working[selected_options]#.fillna('empty').copy()
    mca = prince.MCA(
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    
    mca = mca.fit(mca_data)
    
    chart = mca.plot(
        mca_data,
        x_component=0,
        y_component=1,
        show_column_markers=True,
        show_row_markers=True,
        show_column_labels=False,
        show_row_labels=False
    )
    
    st.write('MCA chart for the selected categorical features')
    st.altair_chart(chart, theme="streamlit",use_container_width=True)

    st.write(mca.eigenvalues_summary)
except:
    st.write('Please select MCA factors')

st.write('When all categorical features are observed, there are obvious trends visible in the scatter plot.')
st.write('With only fuel-type, aspiration, num-of-doors, drive-wheels and make there is no obvious pattern.')
st.write('Variables engine-location, body-style, engine-type and fuel-system add patterns that increase varianse on both axis.')
