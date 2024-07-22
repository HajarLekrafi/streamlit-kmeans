import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le modèle K-means et le préprocesseur
try:
    with open('kmeans_model.pkl', 'rb') as file:
        kmeans_model = pickle.load(file)

    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"Erreur de chargement des fichiers pickle : {e}")
    st.stop()

st.title('K-means Clustering Prediction')

# Uploader un fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :")
    st.write(data.head())

    # Identifier les colonnes catégorielles et numériques
    categorical_columns = data.select_dtypes(include=['object']).columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns

    if len(categorical_columns) > 0:
        st.write(f"Colonnes catégorielles : {categorical_columns.tolist()}")
        st.write(f"Colonnes numériques : {numerical_columns.tolist()}")

        try:
            # Appliquer le prétraitement
            data_preprocessed = preprocessor.transform(data)
            
            # Convertir la matrice creuse en matrice dense si nécessaire
            if hasattr(data_preprocessed, 'toarray'):
                data_preprocessed_dense = data_preprocessed.toarray()
            else:
                data_preprocessed_dense = data_preprocessed

            # Prédiction des clusters
            if st.button('Prédire'):
                try:
                    predictions = kmeans_model.predict(data_preprocessed_dense)
                    
                    # Ajouter les prédictions au DataFrame original
                    data['Cluster'] = predictions
                    
                    # Afficher les résultats de clustering
                    st.write("Résultats de clustering :")
                    st.write(data.head())
                    
                    # Afficher la répartition des clusters
                    st.write("Répartition des clusters :")
                    st.write(data['Cluster'].value_counts())
                    
                    # Afficher les statistiques des clusters
                    cluster_stats = data.groupby('Cluster').agg({
                        'Nb_propositions': ['mean', 'std'],
                        'Ville': ['mean', 'std'],
                        'Courtier': ['mean', 'std'],
                        'Mnt': ['mean', 'std']
                        # Ajoutez d'autres colonnes si nécessaire
                    })
                    st.write("Statistiques des clusters :")
                    st.write(cluster_stats)

                    # Fonction pour tracer les clusters
                    def plot_clusters(data, features, cluster_col):
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(data=data, x=features[0], y=features[1], hue=cluster_col, palette='viridis', s=100, alpha=0.7)
                        plt.title('Visualisation des Clusters')
                        plt.xlabel(features[0])
                        plt.ylabel(features[1])
                        plt.legend(title='Cluster')
                        st.pyplot(plt)
                    
                    # Afficher les clusters (exemple avec les deux premières caractéristiques)
                    if 'Feature1' in data.columns and 'Feature2' in data.columns:
                        plot_clusters(data, ['Feature1', 'Feature2'], 'Cluster')
                    
                    # Afficher les centres des clusters
                    if st.button('Afficher les centres des clusters'):
                        cluster_centers = pd.DataFrame(kmeans_model.cluster_centers_, columns=preprocessor.get_feature_names_out())
                        st.write("Centres des clusters :")
                        st.write(cluster_centers)

                except Exception as e:
                    st.write(f"Une erreur est survenue lors de la prédiction : {e}")
        except Exception as e:
            st.write(f"Une erreur est survenue lors du prétraitement des données : {e}")
    else:
        st.write("Aucune colonne catégorielle trouvée. Assurez-vous que toutes les données sont numériques.")
