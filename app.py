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

st.set_page_config(page_title="Water Quality Dashboard", page_icon="🇷🇼", initial_sidebar_state="expanded")

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


def home_page():
    st.write(f"""# AI based Water Quality Analysis and Forecasting of Vartur Lake Bangalore""", unsafe_allow_html=True)
    
    st.write(f"""<h2>Importance of Urban Lakes</h2>   
    <p>Lakes play a crucial role in urban environments, offering numerous benefits that contribute to the well-being of both the ecosystem and the residents. Here are some key reasons highlighting the importance of lakes in urban areas:</p>
    <p>1.Recreational Opportunities: Urban lakes provide spaces for recreational activities such as boating, fishing, swimming, and picnicking, offering residents a natural escape from the hustle and bustle of city life. These recreational opportunities contribute to improved mental and physical health among urban dwellers.</p>
    <p>2.Aesthetic Value: Lakes enhance the aesthetic appeal of urban landscapes, providing scenic views and green spaces that contribute to a sense of tranquility and beauty within the city. Well-maintained lakes and waterfront areas can become focal points for community gatherings, events, and celebrations.</p> 
    <p>3.Biodiversity and Habitat: Urban lakes support diverse ecosystems and provide habitats for a variety of plant and animal species. They serve as vital breeding grounds for aquatic organisms, birds, and wildlife, contributing to biodiversity conservation and ecological balance within urban environments.</p> 
    <p>4.Water Supply and Quality: Many urban lakes serve as important sources of drinking water for nearby communities. Protecting and preserving these water bodies is essential for maintaining water quality and ensuring a sustainable supply of clean water for urban residents.</p> 
    <p>5.Flood Control and Stormwater Management: Lakes play a crucial role in flood control and stormwater management by absorbing excess rainwater, reducing the risk of flooding, and preventing erosion and soil runoff. Well-designed lakes and wetlands can act as natural buffers against extreme weather events, protecting urban areas from water-related disasters.</p> 
    <p>6.Climate Regulation: Urban lakes help regulate local climate conditions by moderating temperature extremes and reducing the urban heat island effect. The presence of water bodies can lower ambient temperatures, improve air quality, and create microclimates that support vegetation and wildlife.</p> 
    <p>7.Economic Benefits: Lakes contribute to the local economy by attracting tourists, supporting recreational businesses, and enhancing property values in nearby neighborhoods. Waterfront properties are often highly desirable and command premium prices, leading to increased real estate development and investment in urban lakefront areas.</p> 
    <p>8.Cultural and Historical Significance: Many urban lakes have cultural and historical significance, serving as important landmarks or symbols of local identity. They may have historical associations, cultural heritage sites, or traditional uses that contribute to the cultural fabric and heritage of the city.</p> 
    """, unsafe_allow_html=True)

    st.write(f"""<h2>Research Topic</h2> <p>An AI pipeline for the evaluating environmental water ecosystem using scientific associative study of ground and surface water quality</p> """, unsafe_allow_html=True)
    st.write(f"""<h2>Research Objective for Surface and Ground Water</h2> 
	<hr>
	<div style="text-align:center;">
	<table>
	<tr>
		<th width="2%" style="font-size: 140%;">SNo</th>    
		<th width="5%" style="font-size: 140%;">Objective</th>
		<th width="50%" style="font-size: 140%;">Details</th>
	</tr>
	<tr>
		<td>1</td>    
		<td>Site Survey</td>
		<td>At chosen site, perform a preliminary site survey to check the choice of water quality parameter to be measured, to check the logistics of how the measurements will be carried out and to ensure that all the needed parameters have been identified.</td>
  	</tr>
	<tr>
		<td>2</td>    
		<td>Influence of pollution sources</td>
		<td>Explore the influence of pollution sources on the water quality parameters</td>
	</tr>
	<tr>
		<td>3</td>    
		<td>Establish link</td>
		<td>Evaluate whether the Surface Water quality monitoring procedure can be established and linked to any existing programs of government or private agencies like Pollution Control Authorities, Public Health Departments and Universities</td>
	</tr>
 	<tr>
		<td>4</td>    
		<td>Data collection</td>
		<td>Performing the secondary water quality data collection for the chosen site and explore each water quality parameter</td>
	</tr>
	<tr>
		<td>5</td>    
		<td>Statistical analysis</td>
		<td>To perform various statistical analysis on the collected data such as principal component analysis (PCA), regression analysis, and cluster analysis. To carry out correlation and Regression Analysis between physical and chemical parameters of water quality by Canonical Correlation Analysis (CCA)</td>
	</tr>
	<tr>
		<td>6</td>    
		<td>WQI</td>
		<td>To propose and to study water Quality Index so as the complex dataset into a simplified index that is easily understandable by the public.</td>
	</tr>
	<tr>
		<td>7</td>    
		<td>ML Modelling</td>
		<td>Explore and study various Machine Learning Models for Predicting the Water Quality Index</td>
	</tr>
	<tr>
		<td>8</td>    
		<td>Evaluation</td>
		<td>To compare all the employed models in terms of the predictive ability as well as to carry out the error and correlation analysis of the estimated water quality parameters to obtain the most suitable model.</td>
	</tr>
	</table>
	</div>
	<hr>
    """, unsafe_allow_html=True)




def about_page():
    st.write("""<h1>Vartur Lake Bangalore</h1>""", unsafe_allow_html=True)
    st.image(img_vartur_2)
    st.write("""
        <p>Vartur Lake, located in Bangalore, India, is a significant water body that plays a crucial role in the local ecosystem and urban landscape.</p> 
	<p>Here are some key points about Vartur Lake</p>
	<p>Vartur Lake is situated in the eastern part of Bangalore, within the Varthur locality. It lies in close proximity to Whitefield, one of Bangalore's major IT hubs, and is surrounded by residential and commercial developments.</p>
	<p>Vartur Lake covers a considerable area, providing a serene and picturesque setting in the midst of urbanization. The lake's size and features may vary depending on factors such as rainfall, season, and water management practices.</p>
	<p>Vartur Lake serves as an essential habitat for various aquatic plants, birds, and other wildlife species. It supports biodiversity and contributes to the ecological balance of the region, providing nesting grounds for migratory birds and supporting local flora and fauna.</p>
	<p>Challenges and Concerns: Like many urban lakes in Bangalore, Vartur Lake faces challenges such as encroachment, pollution, sewage inflow, and declining water quality. Unregulated urbanization, industrial activities, and inadequate waste management contribute to environmental degradation and threaten the lake's health.</p>
	<p>Conservation Efforts: Various governmental and non-governmental organizations, along with local communities, have initiated efforts to conserve and rejuvenate Vartur Lake. These efforts include lake cleaning drives, restoration projects, and awareness campaigns aimed at promoting sustainable practices and community participation in lake conservation.</p>
	<p>Urban Development Impact: The rapid urban development and population growth in the surrounding areas have put pressure on Vartur Lake's ecosystem. Encroachments, illegal construction, and untreated sewage discharge from nearby settlements pose significant challenges to the lake's long-term sustainability and health.</p>
	<p>Importance for Urban Planning: Vartur Lake serves as a case study for urban planners, environmentalists, and policymakers grappling with issues of water management, urban development, and ecological conservation in rapidly growing cities like Bangalore. Its conservation and sustainable management are integral to ensuring a livable and resilient urban environment for future generations.</p><br>
    """, unsafe_allow_html=True)
    st.image(img_vartur_1)


def model_section():
    st.write("""<h1>Predict Water Quality</h1>
    <p>Enter these values of the parameters to know if the water quality is suitable to drink or not.</p><hr>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        TurbidityNTU = st.number_input(label="Turbidity (NTU)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",
                                       key="test_slider1")
        pH = st.number_input(label="pH", min_value=0.0, max_value=1000.0, step=50.0, format="%f", key="test_slider2")
        
        ConductivityuS = st.number_input(label="Conductivity (uS/cm)", min_value=0.0, max_value=1000.0, step=50.0,
                                         format="%f", key="test_slider3")
    with col2:
        BOD = st.number_input(label="BOD (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider4")
        COD = st.number_input(label="COD (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider5")
        DissolvedOxygen = st.number_input(label="Dissolved Oxygen (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",key="test_slider6")
        
    with col3:
        Chloride = st.number_input(label="Chloride (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",
                                   key="test_slider7")
        Nitrate = st.number_input(label="Nitrate (mg/l)", min_value=0.0, max_value=1000.0, step=50.0, format="%f",
                                  key="test_slider8")
                                  
        st.write("<br>", unsafe_allow_html=True)
        
    predict_button = st.button('  Predict Water Quality  ')

    dataframe = pd.DataFrame({'Turbidity (NTU)': [TurbidityNTU], 'pH': [pH],
                              'Conductivity (uS/cm)': [ConductivityuS],
                              'COD (mg/l)': [COD],
                              'BOD (mg/l)': [BOD], 'Dissolved Oxygen (mg/l)': [DissolvedOxygen],
                              'Chloride (mg/l)': [Chloride], 'Nitrate (mg/l)': [Nitrate],
                             })

    if predict_button:
        model = load_model()
        result = model.predict(dataframe)
        for _ in stqdm(range(50)):
            sleep(0.015)
        if result[0] == 1.0:
            st.error("This Water Quality is Non-Potable")
        else:
            st.success('This Water Quality is Potable')




with st.sidebar:
    st.image(img_rwanda)
    selected = option_menu(
        menu_title=None,
        options=["Home", "Dashboard", "Statistical Analysis", "Correlation Analysis","Outlier Analysis", "Time Series Analysis","Forecasting","Check Water Quality", "About", "Contributors"],
        #icons=["house", "bar_chart", "droplet", "info-circle", "people"],
        styles=css_style
    )

if selected == "Home":
    home_page()
    
if selected == "Dashboard":
    dashboard()

if selected == "Forecasting":
    time_series_forecasting()
    
if selected == "Time Series Analysis":
    time_Series_Analysis()
	
if selected == "Statistical Analysis":
    statistical_Analysis()
    
if selected == "Correlation Analysis":
    correlation_Analysis()
    
if selected == "Outlier Analysis":
    outlier_Analysis()

elif selected == "Check Water Quality":
    model_section()

elif selected == "About":
    about_page()

elif selected == "Contributors":
    contributors_page()