import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import issparse

# Charger le modèle KMeans et le préprocesseur
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

st.title('K-means Clustering Prediction')

# Uploader le fichier CSV
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    # Charger les données
    data = pd.read_csv(uploaded_file)
    st.write("Données chargées :")
    st.write(data.head())
    
    # Vérifier les colonnes et les types de données
    expected_columns = ['Type_pro', 'Nat_pro_concat', 'Nb_propositions', 'Usage', 'Ville', 'Courtier', 'Mnt', 'Jnl']
    
    if all(col in data.columns for col in expected_columns):
        # Convertir les types de données
        try:
            # Convertir les colonnes numériques en float, en remplaçant les valeurs non convertibles par NaN
            for col in ['Nb_propositions', 'Ville', 'Courtier', 'Mnt']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Convertir les colonnes catégorielles en chaînes de caractères
            for col in ['Type_pro', 'Nat_pro_concat', 'Usage', 'Jnl']:
                data[col] = data[col].astype(str)

            # Afficher les types de données après conversion
            st.write("Types de données après conversion :")
            st.write(data.dtypes)
            
            # Gérer les valeurs manquantes (imputation ou suppression)
            data.fillna(method='ffill', inplace=True)
            
            # Prétraitement
            try:
                data_preprocessed = preprocessor.transform(data)
                
                # Convertir les matrices creuses en matrices denses si nécessaire
                if issparse(data_preprocessed):
                    data_preprocessed = data_preprocessed.toarray()
                
                st.write("Données après prétraitement :")
                st.write(pd.DataFrame(data_preprocessed).head())
                st.write("Shape des données prétraitées :")
                st.write(pd.DataFrame(data_preprocessed).shape)
                
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
            st.write("Les colonnes du fichier CSV ne correspondent pas aux colonnes attendues.")
else:
    st.write("Veuillez uploader un fichier CSV.")

