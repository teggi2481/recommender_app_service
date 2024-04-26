
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle as pkl
from keras.models import model_from_json
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
import json

app = Flask("__name__")

DATA_DIR = "./data/"

# First, we'll read the dataset
dataset = pd.read_csv(DATA_DIR+'final_data.csv')
# There's one ' ' value in one of the rows. We have to remove it
dataset['Course'].replace(' ', np.nan, inplace=True)
dataset.dropna(subset=['Course'], inplace=True)
dataset = dataset.reset_index(drop=True)

# Gather unique courses
unique_df = dataset[['University', 'Course', 'IELTS', 'Undergrad', 'work_ex']].drop_duplicates()
unique_df = unique_df.reset_index(drop=True)

unique_courses_df = unique_df[['Course']]
unique_profile_df = unique_df[['IELTS', 'Undergrad', 'work_ex']]

# ---------- load recommender ------------------
# load json and create model
json_file = open(DATA_DIR+'recommender.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
recommender = model_from_json(loaded_model_json)
# load weights into new model
recommender.load_weights(DATA_DIR+"recommender.h5")
print("Loaded recommender model from disk")

# ----------- course encoder -------------------
# load json and create model
json_file = open(DATA_DIR+'course_encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
course_model = model_from_json(loaded_model_json)
# load weights into new model
course_model.load_weights(DATA_DIR+"course_encoder.h5")
print("Loaded course model from disk")

# ---------- profile encoder -------------------
# load json and create model
json_file = open(DATA_DIR+'profile_encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
profile_model = model_from_json(loaded_model_json)
# load weights into new model
profile_model.load_weights(DATA_DIR+"profile_encoder.h5")
print("Loaded profile model from disk")

# ------- one hot encoder and min max scaler -------

course_one_hot_encoder = pkl.load(open(DATA_DIR+"course_one_hot_encoder.pkl", "rb"))
profile_min_max_scaler = pkl.load(open(DATA_DIR+"profile_min_max_scaler.pkl", "rb"))

print("Loaded course one hot encoder & profile min max scaler")


def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def one_hot_encoder(input_value):
    one_hot_vector = course_one_hot_encoder.transform(input_value)
    return one_hot_vector

def min_max_scaler(input_value):
    scaled_value = profile_min_max_scaler.transform(input_value)
    return scaled_value

def concatenate_vectors(v1, v2):
    return np.concatenate((v1, v2), axis=1)

def preprocess_data(course_name_df, profile_df):
    course_name_df['Course'] = course_name_df['Course'].apply(clean_text)
    course_name_np = course_name_df.to_numpy().reshape(course_name_df.shape[0], 1)
    profile_np = profile_df.to_numpy().reshape(profile_df.shape[0], 3)
    course_X_encoded = one_hot_encoder(course_name_np).toarray()
    profile_X_normalized = min_max_scaler(profile_np)
    return course_X_encoded, profile_X_normalized

def list_to_df(list_):
    df = pd.DataFrame([list_])
    return df

def invoke_recommender(course_name_df, profile_df):
    # Preprocess the dataframes
    preprocessed_course, preprocessed_profile = preprocess_data(course_name_df, profile_df)
    # Generate latent vectors
    course_latent_vector = course_model.predict(preprocessed_course)
    profile_latent_vector = profile_model.predict(preprocessed_profile)
    # Concatenate latent vectors
    concatenated_vector = concatenate_vectors(course_latent_vector, profile_latent_vector)
    # Invoke recommender
    output_vector = recommender.predict(concatenated_vector)
    return output_vector

output_vectors = invoke_recommender(unique_courses_df, unique_profile_df)

mapping = {}

for i in range(output_vectors.shape[0]):
    mapping[i] = output_vectors[i]

def fetch_recommendations(output_vector):
    # similarity_score = 0.0
    # similar_idx = None
    similarity_mapping = {}
    # Iterating over latent vectors dictionary
    for idx, vector in mapping.items():
        current_similarity_score = cosine_similarity(vector.reshape(1, vector.shape[0]), output_vector, dense_output=True)
        similarity_mapping[idx] = current_similarity_score.squeeze().item()
    #     if (current_similarity_score > similarity_score):
    #         similarity_score = current_similarity_score
    #         similar_idx = idx


    # Sort the dictionary in ascending order of similarity
    similarity_mapping = {k: v for k, v in sorted(similarity_mapping.items(), key=lambda item: item[1])}
    top_10_similar = list(similarity_mapping.items())[-5:]
    
    results = []
    for i in range(len(top_10_similar)-1, 0, -1):
        uni = unique_df['University'][top_10_similar[i][0]]
        course = unique_df['Course'][top_10_similar[i][0]]
        if ([uni, course] not in results):
            results.append([uni, course])
    return results

def recommend_me(course_name, profile):
    # Convert to Dataframe
    course_name_df = list_to_df(course_name)
    course_name_df = course_name_df.rename(columns={0: 'Course'})
    profile_df = list_to_df(profile)
    output_vector = invoke_recommender(course_name_df, profile_df)
    # Fetch recommendations
    return fetch_recommendations(output_vector)

@app.route("/")
def home():
	return "Welcome to my recommender app!"


@app.route("/recommend", methods=['POST'])
def recommend():
    
    ielts = request.form['IELTS']
    undergrad = request.form['Undergrad']
    workex = request.form['work_ex']
    course = request.form['Course']

    profile = [ielts, undergrad, workex]
    course_name = course
    results = recommend_me(course_name, profile)
    
    response_data = {
        "results": results
        }

    response = app.response_class(
        response=json.dumps(response_data),
        status=200,
        mimetype='application/json'
    )

    return response
if __name__ == "__main__":
    app.run()