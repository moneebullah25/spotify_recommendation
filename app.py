
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

# Load the KNN model
knn_model = joblib.load('knn_model.pkl')
df = pd.read_csv('spotify_data.csv')
df.set_index('track_id', inplace=True)

drop_columns = ['artist_name', 'track_name', 'popularity', 'year']

# Extracting the remaining columns that will be our features
useful_columns = [c for c in df.columns if c not in drop_columns]

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Performing Label Encoding for the Genre column since it is only column which is categorical
df['genre'] = label_encoder.fit_transform(df['genre'])

# Extracting features from the dataframe
features = df[useful_columns]

# Function to recommend songs based on user input
def recommend_songs(user_track_history):
    user_songs = features.loc[user_track_history]
    user_songs = user_songs.drop_duplicates()
    distances, indices = knn_model.kneighbors(user_songs)
    similar_song_indices = indices.flatten()
    recommended_songs = df.iloc[similar_song_indices].drop(user_track_history, errors='ignore')
    return recommended_songs[['artist_name', 'track_name']].head(1)

# Streamlit UI
st.title('Song Recommendation System')

# Input field for user track IDs
user_input = st.text_input('Enter one or more track IDs (separated by commas):')

# Convert user input to a list of track IDs
if user_input:
    user_track_history = user_input.split(',')

    # Display user's input track IDs
    st.write('User Track IDs:', user_track_history)

    # Recommend songs based on user's input track IDs
    recommended_songs = recommend_songs(user_track_history)

    # Display recommended songs
    st.write('Recommended Songs:')
    st.write(recommended_songs)
