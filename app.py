import streamlit as st
import pickle
import pandas as pd
from scipy.sparse import issparse
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

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
st.image(logo_path, width=500, use_column_width=False, output_format='PNG')

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
    "Diagramme en Boîte": st.sidebar.checkbox("Diagramme en Boîte"),
    "Histogramme des valeurs du montant": st.sidebar.checkbox("Histogramme des valeurs du montant"),
    "Diagramme en Violin": st.sidebar.checkbox("Diagramme en Violin"),
    "Histogramme des Villes par Cluster": st.sidebar.checkbox("Histogramme des Villes par Cluster"),
    "Histogramme des Valeurs du Journal par Cluster": st.sidebar.checkbox("Histogramme des Valeurs du Journal par Cluster"),
    "Somme des Montants par Journal": st.sidebar.checkbox("Somme des Montants par Journal"),
    "Distribution des Montants par Année": st.sidebar.checkbox("Distribution des Montants par Année"),
    "Diagramme en Barres Horizontales": st.sidebar.checkbox("Diagramme en Barres Horizontales"),
    "Carte Géographique des Villes": st.sidebar.checkbox("Carte Géographique des Villes"),
    "Nuage de Points des Propositions contre Montant": st.sidebar.checkbox("Nuage de Points des Propositions contre Montant"),
    "Graphique en Lignes des Valeurs du Montant au Fil des Années": st.sidebar.checkbox("Graphique en Lignes des Valeurs du Montant au Fil des Années"),
    "Heatmap des Montants par Ville et Cluster": st.sidebar.checkbox("Heatmap des Montants par Ville et Cluster")
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
                
            # Nettoyer la colonne 'sinistre'
            if 'sinistre' in data.columns:
                data['sinistre'] = pd.to_numeric(data['sinistre'], errors='coerce')
                data['sinistre'].fillna(0, inplace=True)
                data['sinistre'] = data['sinistre'].astype(int)
             

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
                            
                            # Mapper les codes de ville en noms de villes
                            ville_mapping = {
                                10: "AGADIR", 532: "AGHWINIT", 576: "AHFIR", 784: "AIN CHOCK", 783: "AIN SEBAA",
                                22: "AIT MELLOUL", 454: "AIT OURIR", 696: "AKNOUL", 436: "AL HAGGOUNIA", 50: "AL HOCEIMA",
                                24: "ASSA", 32: "ASSILAH", 369: "AZEMMOUR", 402: "AZILAL", 701: "AZROU",
                                108: "BENGUERIR", 116: "BENSLIMANE", 803: "BERKANE", 128: "BERRECHID", 130: "BIR JDID",
                                144: "BOUARFA", 494: "BOUJDOUR", 152: "BOULEMANE", 397: "BOUZNIKA", 160: "CASABLANCA",
                                720: "CHEFCHAOUEN", 491: "CHICHAOUA", 193: "DEMNATE", 821: "DEROUA", 202: "DRIOUCH",
                                246: "ERRACHIDIA", 301: "ESSAOUIRA", 204: "FES", 845: "FOUM ZGUID", 213: "GUELMIM",
                                555: "GUERCIF", 457: "GUIDIMAKA", 220: "IFRANE", 253: "IMINTANOUT", 254: "JERADA",
                                255: "KALAAT MGOUNA", 256: "KALAAT SRAGHNA", 264: "KENITRA", 270: "KHEMISSET",
                                272: "KHENIFRA", 274: "KHOURIBGA", 278: "LAAYOUNE", 312: "LAGOUIRA", 309: "LARACHE",
                                682: "MARRAKECH", 441: "MARTIL", 321: "MECHRAA BEL KSIRI", 661: "MEDIOUNA", 324: "MEKNES",
                                649: "MIDELT", 333: "MISSOUR", 754: "MOHAMMEDIA", 343: "NADOR", 351: "OUARZAZATE",
                                349: "OUAZZANE", 362: "RABAT", 375: "SAFI", 737: "SALE", 399: "SEFROU",
                                710: "SETTAT", 411: "SIDI KACEM", 413: "SIDI SLIMANE", 597: "SKHIRAT", 747: "TAMESNA",
                                417: "TAN TAN", 421: "TANGER", 423: "TAOUNATE", 425: "TAOURIRT", 427: "TAROUDANT",
                                436: "TAZA", 453: "TIZNIT", 456: "TOULAL", 709: "TIFLET", 459: "TILILA",
                                529: "ZAGORA"
                            }
                            data['Ville_Nom'] = data['Ville'].map(ville_mapping).fillna('Inconnue')
                            
                            # Afficher la section sélectionnée dans la barre latérale
                            for option, selected in options.items():
                                if selected:
                                    if option == "Accueil":
                                        st.write("Sélectionnez une option dans la barre de navigation pour afficher les résultats.")
                                    elif option == "Répartition des Clusters":
                                        st.subheader("Répartition des Clusters")
                                        fig_cluster_distribution = px.histogram(data, x='Cluster', color='Cluster',
                                            title="Répartition des Clusters")
                                        st.plotly_chart(fig_cluster_distribution)
                                    elif option == "Diagramme en Boîte":
                                        st.subheader("Diagramme en Boîte des Valeurs du Montant")
                                        fig_box_plot = px.box(data, y='Mnt', color='Cluster',
                                            title="Diagramme en Boîte des Valeurs du Montant par Cluster")
                                        st.plotly_chart(fig_box_plot)
                                    elif option == "Histogramme des valeurs du montant":
                                        st.subheader("Histogramme des Valeurs du Montant")
                                        fig_histogram = px.histogram(data, x='Mnt', color='Cluster',
                                            title="Histogramme des Valeurs du Montant")
                                        st.plotly_chart(fig_histogram)
                                    elif option == "Diagramme en Violin":
                                        st.subheader("Diagramme en Violin des Valeurs du Montant")
                                        fig_violin_plot = px.violin(data, y='Mnt', color='Cluster',
                                            title="Diagramme en Violin des Valeurs du Montant par Cluster")
                                        st.plotly_chart(fig_violin_plot)
                                    elif option == "Histogramme des Villes par Cluster":
                                        st.subheader("Histogramme des Villes par Cluster")
                                        fig_city_histogram = px.histogram(data, x='Ville_Nom', color='Cluster',
                                            title="Histogramme des Villes par Cluster")
                                        st.plotly_chart(fig_city_histogram)
                                    elif option == "Histogramme des Valeurs du Journal par Cluster":
                                        st.subheader("Histogramme des Valeurs du Journal par Cluster")
                                        fig_jnl_histogram = px.histogram(data, x='Jnl', color='Cluster',
                                            title="Histogramme des Valeurs du Journal par Cluster")
                                        st.plotly_chart(fig_jnl_histogram)
                                    elif option == "Somme des Montants par Journal":
                                        st.subheader("Somme des Montants par Journal")
                                        df_sum_jnl = data.groupby('Jnl')['Mnt'].sum().reset_index()
                                        fig_sum_jnl = px.bar(df_sum_jnl, x='Jnl', y='Mnt',
                                            title="Somme des Montants par Journal")
                                        st.plotly_chart(fig_sum_jnl)
                                    elif option == "Distribution des Montants par Année":
                                        st.subheader("Distribution des Montants par Année")
                                        df_distribution = data.groupby('annee')['Mnt'].sum().reset_index()
                                        fig_distribution = px.line(df_distribution, x='annee', y='Mnt',
                                            title="Distribution des Montants par Année")
                                        st.plotly_chart(fig_distribution)
                                    elif option == "Diagramme en Barres Horizontales":
                                        st.subheader("Diagramme en Barres Horizontales des Valeurs du Montant par Ville")
                                        fig_bar_horizontal = px.bar(data, x='Mnt', y='Ville_Nom', color='Cluster',
                                            labels={'Mnt': 'Valeur du Montant', 'Ville_Nom': 'Ville'},
                                            title='Distribution des Valeurs du Montant par Ville')
                                        st.plotly_chart(fig_bar_horizontal)
                                    elif option == "Carte Géographique des Villes":
                                        st.subheader("Carte Géographique des Villes")
                                        fig_map = px.scatter_geo(data, locations='Ville_Nom', locationmode='country names',
                                            color='Cluster', size='Mnt',
                                            title='Carte Géographique des Villes par Cluster')
                                        st.plotly_chart(fig_map)
                                    elif option == "Nuage de Points des Propositions contre Montant":
                                        st.subheader("Nuage de Points des Propositions contre Montant")
                                        fig_scatter = px.scatter(data, x='Nb_propositions', y='Mnt', color='Cluster',
                                            title='Nuage de Points des Propositions contre Montant')
                                        st.plotly_chart(fig_scatter)
                                    elif option == "Graphique en Lignes des Valeurs du Montant au Fil des Années":
                                        st.subheader("Graphique en Lignes des Valeurs du Montant au Fil des Années")
                                        df_yearly = data.groupby('annee')['Mnt'].sum().reset_index()
                                        fig_line = px.line(df_yearly, x='annee', y='Mnt',
                                            title='Valeurs du Montant au Fil des Années')
                                        st.plotly_chart(fig_line)
                                    elif option == "Heatmap des Montants par Ville et Cluster":
                                        st.subheader("Heatmap des Montants par Ville et Cluster")
                                        heatmap_data = data.pivot_table(index='Ville_Nom', columns='Cluster', values='Mnt', aggfunc='sum')
                                        fig_heatmap = px.imshow(heatmap_data, color_continuous_scale='Viridis',
                                            title='Heatmap des Montants par Ville et Cluster')
                                        st.plotly_chart(fig_heatmap)
                                    else:
                                        st.write(f"Option non disponible: {option}")
                        except Exception as e:
                            st.write("Erreur lors de la prédiction des clusters:", e)
        except Exception as e:
            st.write("Erreur lors du prétraitement des données:", e)
else:
    st.write("Veuillez télécharger un fichier CSV pour commencer.")
