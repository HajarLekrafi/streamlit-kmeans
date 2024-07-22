import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

    # Identify categorical and numerical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns

    if len(categorical_columns) > 0:
        st.write(f"Categorical columns: {categorical_columns.tolist()}")
        st.write(f"Numerical columns: {numerical_columns.tolist()}")

        # Define preprocessing for categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(), categorical_columns)
            ]
        )

        try:
            # Apply preprocessing
            data_preprocessed = preprocessor.fit_transform(data)
            
            # Convert sparse matrix to dense matrix
            data_preprocessed_dense = data_preprocessed.toarray()

            st.write("Data after preprocessing:")
            st.write(pd.DataFrame(data_preprocessed_dense).head())

            # Predict button
            if st.button('Predict'):
                try:
                    predictions = kmeans_model.predict(data_preprocessed_dense)
                    st.write("Predictions:")
                    st.write(predictions)
                except Exception as e:
                    st.write(f"An error occurred during prediction: {e}")
        except Exception as e:
            st.write(f"An error occurred during data transformation: {e}")
    else:
        st.write("No categorical columns found. Ensure all data is numeric.")
