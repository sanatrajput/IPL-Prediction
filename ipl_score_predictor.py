import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the pre-trained model from the pickle file
with open('ipl_score_predictor_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Teams for encoding
teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
         'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
         'Delhi Daredevils', 'Sunrisers Hyderabad']

# Initialize LabelEncoder for team encoding
le = LabelEncoder()
le.fit(teams)

# OneHotEncoder for team encoding (assuming it was used during training)
onehotencoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
team_encoded = onehotencoder.fit_transform(np.array(teams).reshape(-1, 1))

# Function to create the feature vector in the same format as used during training
def create_feature_vector(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5):
    # Encode teams using LabelEncoder
    batting_team_encoded = le.transform([batting_team])[0]
    bowling_team_encoded = le.transform([bowling_team])[0]
    
    # One-hot encode the teams
    batting_team_onehot = onehotencoder.transform(np.array([batting_team]).reshape(-1, 1)).flatten()
    bowling_team_onehot = onehotencoder.transform(np.array([bowling_team]).reshape(-1, 1)).flatten()
    
    # Prepare the feature vector (we will need to match the order and number of features from the training)
    feature_vector = np.array([
        *batting_team_onehot, *bowling_team_onehot, runs, wickets, overs, runs_last_5, wickets_last_5
    ])
    
    # Return the feature vector in the same format as expected by the model
    return feature_vector.reshape(1, -1)

# Function to make predictions
def score_predict(feature_vector):
    # Make prediction using the model
    pred = model.predict(feature_vector)
    return int(round(pred[0]))

# Streamlit UI
st.title("IPL Score Predictor")

# Assign unique keys to selectboxes
batting_team = st.selectbox('Select Batting Team', teams, key='batting_team')
bowling_team = st.selectbox('Select Bowling Team', teams, key='bowling_team')

# Input fields for other features
runs = st.number_input("Enter Runs", min_value=0, max_value=500, step=1)
wickets = st.number_input("Enter Wickets", min_value=0, max_value=10, step=1)
overs = st.number_input("Enter Overs", min_value=0.1, max_value=50.0, step=0.1)
runs_last_5 = st.number_input("Enter Runs in Last 5 Overs", min_value=0, max_value=100, step=1)
wickets_last_5 = st.number_input("Enter Wickets in Last 5 Overs", min_value=0, max_value=5, step=1)

if st.button("Predict Score"):
    # Create the feature vector for prediction
    feature_vector = create_feature_vector(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5)
    
    # Make prediction
    predicted_score = score_predict(feature_vector)
    st.write(f"Predicted Score for {batting_team} vs {bowling_team}: {predicted_score}")
