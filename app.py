import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

    # Check if the model is a KMeans model
    if not hasattr(kmeans_model, 'predict'):
        st.write("Error: The loaded model does not have a 'predict' method.")
    else:
        # Ensure the data is in the correct format (e.g., numeric)
        if data.select_dtypes(include=[np.number]).empty:
            st.write("Error: The uploaded data does not contain numeric columns.")
        else:
            # Assuming the model was trained on standardized data
            # Apply the same transformations if necessary
            scaler = StandardScaler()
            try:
                data_scaled = scaler.fit_transform(data)  # Replace with proper scaler if needed
                st.write("Data after scaling:")
                st.write(pd.DataFrame(data_scaled, columns=data.columns).head())

                # Predict button
                if st.button('Predict'):
                    try:
                        predictions = kmeans_model.predict(data_scaled)
                        st.write("Predictions:")
                        st.write(predictions)
                    except Exception as e:
                        st.write(f"An error occurred during prediction: {e}")
            except Exception as e:
                st.write(f"An error occurred during data transformation: {e}")
