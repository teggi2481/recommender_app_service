import stqdm
import os
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
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image

subscription_key = "613104a1a93e4983b9aa943e9bf46b92"
endpoint = "https://pralhad-computer-vision-ws.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

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

def file_selector(folder_path='./images/'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def home_page():
    st.write(f"""# Recommendation System""", unsafe_allow_html=True)

    st.write(f"""#### Study Material Recommendation Systems based on Study Patterns and Cognitive learning""", unsafe_allow_html=True)
    st.write(f"""<p>This project takes the log files of user as initial input and extracts their study patterns based on their online activity. The recommendation 
    generator takes those features as input, tries to find similar users based comparing the study patterns and recommends courses they completed.
    </p>""", unsafe_allow_html=True)

    st.write('Select the User Image ')
    filename = file_selector()
    st.write('You selected `%s`' % filename)
    image = open(filename, "rb")  
    detect_objects_results = computervision_client.detect_objects_in_stream(image)
    tags_result = computervision_client.tag_image_in_stream(image)
    for tag in tags_result.tags:
        st.write(tags.name)
    st.write('Provide an index between 0 to ',max,' to choose a USER ID ')
    input_userid = st.number_input(label="", min_value=0, max_value=100, step=1, key="test_slider1")
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
    st.write(f"""<p>This project takes the log files of user as initial input and extracts their study patterns based on their online activity. The recommendation 
    generator takes those features as input, tries to find similar users based comparing the study patterns and recommends courses they completed.
    </p>""", unsafe_allow_html=True)
    st.write("""#### Developed by Pralhad P. Teggi""", unsafe_allow_html=True)
    st.write("""#### DStudent ID - 224231596""", unsafe_allow_html=True)
    st.write("""#### Email - pralhad2481@gmail.com""", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home","About"],
        icons=["house"],
        styles=css_style
    )

if selected == "Home":
    home_page()
if selected == "About":
    about_page()
    
