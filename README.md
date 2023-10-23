# Automobile_app
Streamlit app for EDA and modeling of kaggle's automobile dataset

This folder contains the technical assessment task for BlackPeack. 
All subtasks are organized as pages in one streamlit app. 
To load the app: 
1. Pull the folder to drive
2. Change the project_path to the path to the folder destination. This has to be done in the main script only, Automobile_sales_modeling.py
3. To initiate the streampit app: install all requirements, open anaconda prompt and run **streamlit run folder/Automobile_sales_modeling.py**

To differ from browser tab, the streamlit app has the followning 6 app tabs: Automobile_sales_modeling, EDA, MCA, Modeling, Price prediction and Deployment plan. 
Note: session_state is instantated in the first app tab - Automobile_sales_modeling. This is a streamlit constraint.
If the browser is reloaded while on another app tab, you need to switch back to the first app tab to reinstantiate the session_state.

Multiple pickles are saved in the pickles directory - data format, model and the data itself. They are part of the app.
