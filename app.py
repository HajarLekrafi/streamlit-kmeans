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

# Lire le fichier CSS
css_file_path = 'style.css'
with open(css_file_path) as f:
    css = f.read()

# Inclure le CSS dans la page
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Afficher le logo
logo_path = 'logo.png'
st.image(logo_path, width=200)

# Titre de l'application
st.markdown('<h1 class="title">K-means Clustering Prediction</h1>', unsafe_allow_html=True)

# Description
st.markdown('<p class="description">Téléchargez un fichier CSV pour prédire les clusters à l\'aide du modèle KMeans.</p>', unsafe_allow_html=True)

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
                            
                            
                             # Afficher la répartition des clusters avec labels
                            st.write("Répartition des clusters avec labels :")
                            cluster_distribution = data.groupby('Cluster_Label').size().reset_index(name='Count')
                            st.write(cluster_distribution)
                            
                            # Déterminer si 'Sinistre' ou 'sinistre' est présent et afficher les prédictions
                            sinistre_col = None
                            if 'Sinistre' in data.columns:
                                sinistre_col = 'Sinistre'
                            elif 'sinistre' in data.columns:
                                sinistre_col = 'sinistre'
                            
                            if sinistre_col:
                                st.write("Toutes les prédictions avec labels :")
                                st.write(data[[sinistre_col, 'Cluster']])
                            else:
                                st.write("Toutes les prédictions avec labels :")
                                st.write(data[['Cluster']])
                            
                            # Afficher des graphiques interactifs
                            st.subheader("Répartition des Clusters")
                            cluster_distribution = data['Cluster'].value_counts().reset_index()
                            cluster_distribution.columns = ['Cluster', 'Count']
                            fig = px.bar(cluster_distribution, x='Cluster', y='Count', 
                                         labels={'Cluster': 'Cluster', 'Count': 'Nombre d\'Occurrences'},
                                         title='Répartition des Clusters')
                            st.plotly_chart(fig)

                            st.subheader("Analyse des Données")
                            fig_scatter = px.scatter(data, x='Nb_propositions', y='Mnt', color='Cluster', labels={'Nb_propositions': 'Nombre de Propositions', 'Mnt': 'Montant'},
                                                     title='Analyse des Données par Cluster')
                            st.plotly_chart(fig_scatter)
                            
                            # Ajouter un diagramme en boîte
                            st.subheader("Diagramme en Boîte")
                            fig_box = px.box(data, x='Cluster', y='Nb_propositions', 
                                             title='Répartition des Nombres de Propositions par Cluster',
                                             labels={'Cluster': 'Cluster', 'Nb_propositions': 'Nombre de Propositions'})
                            st.plotly_chart(fig_box)
                            
                            # Ajouter un histogramme
                            st.subheader("Histogramme")
                            fig_hist = px.histogram(data, x='Mnt', color='Cluster',
                                                    title='Répartition des Montants par Cluster',
                                                    labels={'Mnt': 'Montant', 'Cluster': 'Cluster'})
                            st.plotly_chart(fig_hist)

                            # Ajouter un diagramme en Violin
                            st.subheader("Diagramme en Violin")
                            fig_violin = px.violin(data, y='Mnt', color='Cluster', box=True, 
                                                  title='Distribution des Montants par Cluster',
                                                  labels={'Mnt': 'Montant', 'Cluster': 'Label du Cluster'})
                            st.plotly_chart(fig_violin)

                            # Ajouter un histogramme des Villes par Cluster
                            if 'Ville' in data.columns:
                                st.subheader("Histogramme des Villes par Cluster")
                                ville_cluster = data.groupby(['Ville', 'Cluster']).size().reset_index(name='Count')
                                fig_ville_cluster = px.histogram(ville_cluster, x='Ville', y='Count', color='Cluster',
                                                                 labels={'Ville': 'Ville', 'Count': 'Nombre d\'Occurrences'},
                                                                 title='Répartition des Villes par Cluster')
                                st.plotly_chart(fig_ville_cluster)
                            
                            # Ajouter un histogramme des Courtiers par Ville
                            if 'Courtier' in data.columns and 'Ville' in data.columns:
                                st.subheader("Histogramme des Courtiers par Ville")
                                courtier_ville = data.groupby(['Ville', 'Courtier']).size().reset_index(name='Count')
                                fig_courtier_ville = px.histogram(courtier_ville, x='Ville', y='Count', color='Courtier',
                                                                  labels={'Ville': 'Ville', 'Count': 'Nombre de Courtiers'},
                                                                  title='Répartition des Courtiers par Ville')
                                st.plotly_chart(fig_courtier_ville)

                    
                            
                        except Exception as e:
                            st.write(f"Erreur lors de la prédiction des clusters : {e}")
                except Exception as e:
                    st.write(f"Erreur lors du prétraitement des données : {e}")
            else:
                st.write("Le préprocesseur n'est pas ajusté. Veuillez ajuster le préprocesseur avant de l'utiliser.")
        except Exception as e:
            st.write(f"Erreur lors de la conversion des types de données : {e}")
    else:
        st.write("Les colonnes du fichier CSV ne correspondent pas aux colonnes attendues.")
else:
    st.write("Veuillez uploader un fichier CSV.")
