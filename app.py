
import streamlit as st
import pickle
import pandas as pd

# Load the K-means model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

st.title('K-means Clustering Prediction')

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Predict button
    if st.button('Predict'):
        predictions = kmeans_model.predict(data)
        st.write("Predictions:")
        st.write(predictions)
    