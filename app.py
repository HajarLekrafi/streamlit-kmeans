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

# Exemple de CSS directement dans le code
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
        }
        .description {
            font-size: 18px;
            color: #333;
        }
        .btn-custom {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
    </style>
    <div class="title">K-means Clustering Prediction</div>
    <div class="description">Téléchargez un fichier CSV pour prédire les clusters.</div>
""", unsafe_allow_html=True)

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
            # Convertir les colonnes numériques en float, en remplaçant les valeurs non convertibles par NaN
            for col in ['Nb_propositions', 'Ville', 'Courtier', 'Mnt']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Convertir les colonnes catégorielles en chaînes de caractères
            for col in ['Type_pro', 'Nat_pro_concat', 'Usage', 'Jnl']:
                data[col] = data[col].astype(str)

            # Gérer les valeurs manquantes (imputation ou suppression)
            if hasattr(preprocessor, 'transform'):
                # Prétraitement
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
                                0: "Faible Valeur, Faible Nombre de Propositions, Localisations Eparses",
                                1: "Faible Valeur, Faible Nombre de Propositions, Localisations Mixtes",
                                2: "Haute Valeur, Grand Nombre de Propositions, Localisations Eparses",
                                3: "Valeur Moyenne, Très Faible Nombre de Propositions, Localisations Concentrees"
                            }
                            data['Cluster_Label'] = data['Cluster'].map(labels)
                            
                            # Afficher la répartition des clusters
                            st.write("Répartition des clusters :")
                            cluster_distribution = data['Cluster'].value_counts()
                            st.write(cluster_distribution)
                            
                            # Afficher toutes les prédictions avec labels
                            st.write("Toutes les prédictions avec labels :")
                            st.write(data[['Cluster', 'Cluster_Label']])
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
