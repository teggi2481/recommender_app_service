import stqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import time
from datetime import datetime
from PIL import Image
from time import sleep
import streamlit as st
from stqdm import stqdm
from streamlit_option_menu import option_menu
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from generator import recommendationGenerator

st.set_page_config(page_title="Recommendation System", page_icon="🇷🇼", initial_sidebar_state="expanded")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
css_style = {
    "icon": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B"},
}

features,data = recommendationGenerator.load_data(recommendationGenerator, datapath = './data/features_sample.csv')
users = data['userID'].unique()
max = users.shape[0] -1
recomm = None
def home_page():
    st.write(f"""# Recommendation System""", unsafe_allow_html=True)
    
    
    input_userid = st.number_input(label="Insert an index between 0 to {{max}} to choose a USER ID", min_value=0, max_value=100, step=1, key="test_slider1")
    n = st.number_input(label="Number of Recommendations", min_value=0, max_value=100, step=1, key="test_slider2")
    

    predict_button = st.button('Recommend')

    if predict_button:
            #Getting the userID from the user Index
	    users = data['userID'].unique()
	    userID = int(users[input_userid])

	    #Running the model, generating recommendations and passing the list to the HTML page
	    model = recommendationGenerator(userID, n)
	    recomm = model.generate_recommendations(features,data)
	    if recomm != None :
            	st.write(f"""#### Recommendation System Results""", unsafe_allow_html=True)
        	st.write(recomm)
	    
def results():
    st.write(f"""# Recommendation System Results""", unsafe_allow_html=True)
    

def about_page():
    st.write("""<h1>Recommendation System</h1>""", unsafe_allow_html=True)
    
    st.write("""
        <p>Vartur Lake, located in Bangalore, India, is a significant water body that plays a crucial role in the local ecosystem and urban landscape.</p> 
    """, unsafe_allow_html=True)
    


with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home"],
        icons=["house"],
        styles=css_style
    )

if selected == "Home":
    home_page()
if selected == "Results":
    results()
    
