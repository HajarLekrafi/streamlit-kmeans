import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import issparse
from sklearn.impute import SimpleImputer
import plotly.express as px

# Charger le modèle KMeans et le préprocesseur
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Lire le CSS depuis le fichier
with open('style.css', 'r') as file:
    css = file.read()

# Inclure le CSS dans la page
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Afficher le logo
logo_path = 'logo.png'
st.image(logo_path, width=200, use_column_width=False, output_format='PNG')

# Titre de l'application
st.markdown('<h1 class="title">K-means Clustering Prediction</h1>', unsafe_allow_html=True)

# Description
st.markdown('<p class="description">Téléchargez un fichier CSV pour prédire les clusters à l\'aide du modèle KMeans.</p>', unsafe_allow_html=True)

# loader HTML
loader_html = """
<div class="loader">
<div class="loader-square"></div>
<div class="loader-square"></div>
<div class="loader-square"></div>
<div class="loader-square"></div>
<div class="loader-square"></div>
<div class="loader-square"></div>
<div class="loader-square"></div>
</div>
"""

# Inclure le spinner dans la page
st.markdown(loader_html, unsafe_allow_html=True)

# Sidebar for navigation with custom checkboxes
st.sidebar.header("Navigation")
options = {
    "Accueil": st.sidebar.checkbox("Accueil"),
    "Répartition des Clusters": st.sidebar.checkbox("Répartition des Clusters"),
    "Analyse des Données": st.sidebar.checkbox("Analyse des Données"),
    "Diagramme en Boîte": st.sidebar.checkbox("Diagramme en Boîte"),
    "Histogramme": st.sidebar.checkbox("Histogramme"),
    "Diagramme en Violin": st.sidebar.checkbox("Diagramme en Violin"),
    "Histogramme des Villes par Cluster": st.sidebar.checkbox("Histogramme des Villes par Cluster"),
    "Histogramme des Courtiers par Ville": st.sidebar.checkbox("Histogramme des Courtiers par Ville"),
    "Histogramme des Valeurs du Journal par Cluster": st.sidebar.checkbox("Histogramme des Valeurs du Journal par Cluster")
}

# Uploader le fichier CSV
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    # Charger les données
    data = pd.read_csv(uploaded_file)
    st.write("<div class='data-table'>Données chargées :</div>", unsafe_allow_html=True)
    st.write(data.head())
    
    # Vérifier les colonnes et les types de données
    expected_columns = ['Type_pro', 'Nat_pro_concat', 'Nb_propositions', 'Usage', 'Ville', 'Courtier', 'Mnt', 'Jnl']
    
    if all(col in data.columns for col in expected_columns):
        try:
            # Convertir les colonnes numériques en float
            for col in ['Nb_propositions', 'Ville', 'Courtier', 'Mnt']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Convertir les colonnes catégorielles en chaînes de caractères
            for col in ['Type_pro', 'Nat_pro_concat', 'Usage', 'Jnl']:
                data[col] = data[col].astype(str)

            # Imputer les valeurs manquantes
            numeric_cols = ['Nb_propositions', 'Ville', 'Courtier', 'Mnt']
            imputer = SimpleImputer(strategy='mean')
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            
            # Gérer les valeurs manquantes pour les colonnes catégorielles
            for col in ['Type_pro', 'Nat_pro_concat', 'Usage', 'Jnl']:
                data[col].fillna('Unknown', inplace=True)

            # Prétraitement
            if hasattr(preprocessor, 'transform'):
                try:
                    data_preprocessed = preprocessor.transform(data)
                    
                    # Convertir les matrices creuses en matrices denses si nécessaire
                    if issparse(data_preprocessed):
                        data_preprocessed = data_preprocessed.toarray()
                    
                    # Prédiction des clusters
                    if st.button('Prédire les Clusters'):
                        try:
                            predictions = kmeans_model.predict(data_preprocessed)
                            data['Cluster'] = predictions
                            
                            # Ajouter les labels des clusters
                            labels = {
                                0: "Faible Valeur du montant, Faible Nombre de Propositions, Localisations Eparses",
                                1: "Faible Valeur du montant, Faible Nombre de Propositions, Localisations Mixtes",
                                2: "Haute Valeur du montant, Grand Nombre de Propositions, Localisations Eparses",
                                3: "Valeur du montant Moyenne, Très Faible Nombre de Propositions, Localisations Concentrees"
                            }
                            data['Cluster_Label'] = data['Cluster'].map(labels)
                            
                            # Afficher la section sélectionnée dans la barre latérale
                            for option, selected in options.items():
                                if selected:
                                    if option == "Accueil":
                                        st.write("Sélectionnez une option dans la barre de navigation pour afficher les résultats.")
                                        
                                    elif option == "Répartition des Clusters":
                                        st.subheader("Répartition des Clusters")
                                        cluster_distribution = data['Cluster_Label'].value_counts().reset_index()
                                        cluster_distribution.columns = ['Cluster_Label', 'Count']
                                        fig = px.bar(cluster_distribution, x='Cluster_Label', y='Count',
                                                    labels={'Cluster_Label': 'Cluster', 'Count': 'Nombre d\'Occurrences'},
                                                    title='Répartition des Clusters')
                                        st.plotly_chart(fig)
                                        
                                    elif option == "Analyse des Données":
                                        st.subheader("Analyse des Données")
                                        st.write(data.describe())
                                        
                                    elif option == "Diagramme en Boîte":
                                        st.subheader("Diagramme en Boîte")
                                        fig_box = px.box(data, y='Mnt', color='Cluster',
                                                         labels={'Mnt': 'Valeur du Montant', 'Cluster': 'Cluster'},
                                                         title='Diagramme en Boîte des Valeurs du Montant par Cluster')
                                        st.plotly_chart(fig_box)
                                        
                                    elif option == "Histogramme":
                                        st.subheader("Histogramme")
                                        hist_fig = px.histogram(data, x='Mnt', color='Cluster',
                                                                labels={'Mnt': 'Valeur du Montant', 'Cluster': 'Cluster'},
                                                                title='Histogramme des Valeurs du Montant par Cluster')
                                        st.plotly_chart(hist_fig)
                                        
                                    elif option == "Diagramme en Violin":
                                        st.subheader("Diagramme en Violin")
                                        fig_violin = px.violin(data, y='Mnt', color='Cluster',
                                                              labels={'Mnt': 'Valeur du Montant', 'Cluster': 'Cluster'},
                                                              title='Diagramme en Violin des Valeurs du Montant par Cluster')
                                        st.plotly_chart(fig_violin)
                                        
                                    elif option == "Histogramme des Villes par Cluster":
                                        st.subheader("Histogramme des Villes par Cluster")
                                        ville_cluster = data.groupby(['Ville', 'Cluster']).size().reset_index(name='Count')
                                        fig_ville_cluster = px.histogram(ville_cluster, x='Ville', y='Count', color='Cluster',
                                                                         labels={'Ville': 'Ville', 'Count': 'Nombre d\'Occurrences'},
                                                                         title='Répartition des Villes par Cluster')
                                        st.plotly_chart(fig_ville_cluster)
                                        
                                    elif option == "Histogramme des Courtiers par Ville":
                                        if 'Courtier' in data.columns and 'Ville' in data.columns:
                                            st.subheader("Histogramme des Courtiers par Ville")
                                            courtier_ville = data.groupby(['Ville', 'Courtier']).size().reset_index(name='Count')
                                            fig_courtier_ville = px.histogram(courtier_ville, x='Ville', y='Count', color='Courtier',
                                                                              labels={'Ville': 'Ville', 'Count': 'Nombre de Courtiers'},
                                                                              title='Répartition des Courtiers par Ville')
                                            st.plotly_chart(fig_courtier_ville)
                                            
                                    elif option == "Histogramme des Valeurs du Journal par Cluster":
                                        if 'Jnl' in data.columns:
                                            st.subheader("Histogramme des Valeurs du Journal par Cluster")
                                            fig_jnl = px.histogram(data, x='Jnl', color='Cluster', 
                                                                   labels={'Jnl': 'Valeur du Journal', 'Cluster': 'Cluster'},
                                                                   title='Distribution des Valeurs du journal par Cluster')
                                            st.plotly_chart(fig_jnl)
                                
                            # Déterminer si 'Sinistre' ou 'sinistre' est présent et afficher les prédictions
                            sinistre_col = None
                            if 'Sinistre' in data.columns:
                                sinistre_col = 'Sinistre'
                            elif 'sinistre' in data.columns:
                                sinistre_col = 'sinistre'
                            
                            if sinistre_col:
                                st.write("<div class='data-table'>Toutes les prédictions avec labels :</div>", unsafe_allow_html=True)
                                st.write(data[[sinistre_col, 'Cluster', 'Cluster_Label']])
                            else:
                                st.write("<div class='data-table'>Toutes les prédictions avec labels :</div>", unsafe_allow_html=True)
                                st.write(data[['Cluster', 'Cluster_Label']])
                                
                        except Exception as e:
                            st.write(f"Erreur lors de la prédiction des clusters : {e}")
                except Exception as e:
                    st.write(f"Erreur lors du prétraitement des données : {e}")
            else:
                st.write("Le préprocesseur n'est pas ajusté. Veuillez ajuster le préprocesseur pour les données.")
        except Exception as e:
            st.write(f"Erreur lors de la préparation des données : {e}")
    else:
        st.write("Le fichier CSV ne contient pas toutes les colonnes nécessaires.")
