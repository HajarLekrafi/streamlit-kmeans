import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.sparse import issparse

# Charger le modèle KMeans et le préprocesseur
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

st.title('K-means Clustering Prediction')

# Uploader le fichier CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Charger les données
    data = pd.read_csv(uploaded_file)
    st.write("Données chargées :")
    st.write(data.head())
    
    # Prétraitement
    try:
        data_preprocessed = preprocessor.transform(data)
        
        # Convertir les matrices creuses en matrices denses si nécessaire
        if issparse(data_preprocessed):
            data_preprocessed = data_preprocessed.toarray()
        
        st.write("Données après prétraitement :")
        st.write(pd.DataFrame(data_preprocessed).head())
        
        # Afficher les centres des clusters
        if st.button('Afficher Centres des Clusters'):
            try:
                centers = kmeans_model.cluster_centers_
                st.write("Centres des clusters :")
                st.write(centers)
            except Exception as e:
                st.write(f"Erreur lors de l'affichage des centres des clusters : {e}")
        
        # Prédiction des clusters
        if st.button('Prédire les Clusters'):
            try:
                predictions = kmeans_model.predict(data_preprocessed)
                data['Cluster'] = predictions
                
                # Afficher la répartition des clusters
                st.write("Répartition des clusters :")
                cluster_distribution = data['Cluster'].value_counts()
                st.write(cluster_distribution)
                
                # Afficher les 10 premières prédictions
                st.write("Prédictions des premiers échantillons :")
                st.write(predictions[:10])
            except Exception as e:
                st.write(f"Erreur lors de la prédiction des clusters : {e}")
    except Exception as e:
        st.write(f"Erreur lors du prétraitement des données : {e}")
else:
    st.write("Veuillez uploader un fichier CSV.")
