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
                                        cluster_distribution = data['Cluster'].value_counts().reset_index()
                                        cluster_distribution.columns = ['Cluster', 'Count']
                                        fig = px.bar(cluster_distribution, x='Cluster', y='Count',
                                                    labels={'Cluster': 'Cluster', 'Count': 'Nombre d\'Occurrences'},
                                                    title='Répartition des Clusters')
                                        st.plotly_chart(fig)
                                        
                                        # Afficher la répartition des clusters avec labels
                                        st.write("Répartition des clusters avec labels :")
                                        cluster_distribution_labels = data.groupby('Cluster_Label').size().reset_index(name='Count')
                                        st.write(cluster_distribution_labels)
                                        
                                    elif option == "Diagramme en Boîte":
                                        st.subheader("Diagramme en Boîte")
                                        fig_box = px.box(data, y='Mnt', color='Cluster',
                                                         labels={'Mnt': 'Valeur du Montant', 'Cluster': 'Cluster'},
                                                         title='Diagramme en Boîte des Valeurs du Montant par Cluster')
                                        st.plotly_chart(fig_box)
                                        
                                    elif option == "Histogramme des valeurs du montant":
                                        st.subheader("Histogramme des valeurs du montant")
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
    
                                        # Grouper par 'Cluster' et 'Ville' et compter les occurrences
                                        ville_cluster = data.groupby(['Cluster', 'Ville']).size().reset_index(name='Count')
                                        
                                        # Trouver la ville avec le maximum d'occurrences pour chaque cluster
                                        idx_max_villes = ville_cluster.groupby('Cluster')['Count'].idxmax()
                                        villes_max = ville_cluster.loc[idx_max_villes].reset_index(drop=True)
                                        
                                        # Afficher le DataFrame des villes les plus fréquentes par cluster
                                        st.write("<div class='data-table'>Villes les plus fréquentes par Cluster :</div>", unsafe_allow_html=True)
                                        st.write(villes_max)
                                        
                                        # Créer et afficher le graphique
                                        fig_ville_cluster = px.bar(villes_max, x='Cluster', y='Count', color='Ville',
                                                                labels={'Cluster': 'Cluster', 'Count': 'Nombre d\'Occurrences', 'Ville': 'Ville'},
                                                                title='Ville la plus fréquente par Cluster')
                                        st.plotly_chart(fig_ville_cluster)

                                        
                                    elif option == "Histogramme des Valeurs du Journal par Cluster":
                                        if 'Jnl' in data.columns:
                                            st.subheader("Histogramme des Valeurs du Journal par Cluster")
                                            fig_jnl = px.histogram(data, x='Jnl', color='Cluster', 
                                                                   labels={'Jnl': 'Valeur du Journal', 'Cluster': 'Cluster'},
                                                                   title='Distribution des Valeurs du journal par Cluster')
                                            st.plotly_chart(fig_jnl)
                                            
                                            
                                    elif option == "Histogramme des Villes par Cluster":
                                        # Ajouter la colonne des noms des villes si elle n'existe pas déjà
                                        if 'Ville_Nom' not in data.columns:
                                            ville_mapping = {
                                                10: "AGADIR", 532: "AGHWINIT", 576: "AHFIR", 784: "AIN CHOCK", 783: "AIN SEBAA",
                                                22: "AIT MELLOUL", 454: "AIT OURIR", 696: "AKNOUL", 436: "AL HAGGOUNIA", 50: "AL HOCEIMA",
                                                433: "AL MARSA (LAYOUNE PLAGE)", 264: "AMGALA", 531: "AWSERD", 176: "AZEMMOUR", 643: "AZILAH",
                                                70: "AZILAL", 425: "AZROU", 615: "BEN AHMED", 201: "BEN GUERIR", 781: "BEN M'SIK",
                                                110: "BEN SLIMANE", 90: "BENI MELLAL", 575: "BERKANE", 621: "BERRCHID", 536: "BIR ANZARANE",
                                                542: "BIR GANDOUZ", 312: "BOUARFA", 130: "BOUJDOUR", 431: "BOUKRAA", 140: "BOULEMANE",
                                                791: "BOUSKOURA", 121: "BOUZNIKA", 789: "CASA ANFA", 780: "CASABLANCA", 150: "CHEFCHAOUEN",
                                                601: "CHEMAIA", 465: "CHICHAOUA", 530: "DAKHLA", 792: "DAR BOUAZZA", 435: "DARWA",
                                                432: "DCHEIRA (LAAYOUNE)", 11: "DCHIRA (AGADIR)", 782: "DERB SULTAN", 501: "DRIOUCH",
                                                544: "EL ARGOUB", 622: "EL GARA", 491: "EL HAJEB", 170: "EL JADIDA", 190: "EL KALAA DES SRAGHNA",
                                                218: "ERFOUD", 210: "ERRACHIDIA", 240: "ESSAOUIRA", 260: "ES-SMARA", 900: "ETRANGER",
                                                682: "FAM AL HISSN", 270: "FES", 310: "FIGUIG", 101: "FKIH BEN SALAH", 726: "FNIDEK",
                                                439: "FOUM EL OUED", 223: "GOULMIMA", 537: "GUEBILAT EL FOULA", 320: "GUELMIM", 135: "GUELTAT ZEMMOUR",
                                                703: "GUERCIF", 266: "HAWZA", 761: "IFNI", 420: "IFRANE", 440: "IKHFENNIR", 545: "IMILILI",
                                                471: "IMI-N-TANOUTE", 21: "INEZGANE", 262: "JDIRIA", 582: "JERADA", 131: "JRIFIA", 104: "KASBAT TADLA",
                                                330: "KENITRA", 186: "KHEMIS ZMAMRA", 360: "KHEMISSET", 380: "KHENIFRA", 400: "KHOURIBGA",
                                                731: "KSAR EL KEBIR", 430: "LAAYOUNE", 547: "LAGOUIRA", 735: "LARACHE", 450: "MARRAKECH",
                                                727: "MARTIL", 336: "MECHRA BEL KSIRI", 728: "MEDIEK", 793: "MEDIOUNA", 480: "MEKNES",
                                                860: "MELILLIA", 391: "MIDELT", 538: "MIJIK", 144: "MISSOUR", 787: "MOHAMMEDIA", 653: "M'SIED",
                                                500: "NADOR", 794: "NOUACER", 550: "OUARZAZATE", 341: "OUAZZANE", 695: "OUED AMLIL", 411: "OUED ZEM",
                                                570: "OUJDA", 539: "OUM DREYGA", 810: "RABAT", 235: "RISSANI", 368: "ROMMANI", 590: "SAFI",
                                                815: "SALE", 850: "SEBTA", 281: "SEFROU", 610: "SETTAT", 263: "SIDI AHMED LAAROUSSI",
                                                179: "SIDI BENNOUR", 346: "SIDI KACEM", 349: "SIDI SLIMANE", 332: "SIDI YAHIA EL GHARB",
                                                124: "SIDI YAHIA ZAIRE", 821: "SKHIRATE", 353: "SOUK LARBAA EL RHARB", 765: "TAFRAOUT", 438: "TAH",
                                                650: "TAN TAN", 640: "TANGER", 660: "TAOUNATE", 585: "TAOURIRT", 437: "TARFAYA", 61: "TARGUISTE",
                                                41: "TAROUDANT", 680: "TATA", 690: "TAZA", 825: "TEMARA", 720: "TETOUAN", 268: "TFARITI", 533: "TICHLA",
                                                374: "TIFLET", 559: "TINEGHIR", 750: "TIZNIT", 607: "YOUSSOUFIA", 565: "ZAGORA", 519: "ZAIO",
                                                534: "ZOUG", 412: "BOUJAAD", 28: "OULED TAIMA", 937: "NOUAKCHOTT", 81: "DEMNATE",
                                                434: "OUED EDDAHAB", 151: "Bab Taza", 587: "SIDI MELLOUK", 822: "Ain Aouda", 523: "BEN TAIB",
                                                57: "IMZOUREN", 629: "HAD SOUALEM", 507: "ZGHENGHEN", 812: "AIN ATIK", 347: "JORF ELMELHA",
                                                194: "EL ATTAOUIA", 461: "AMIZMIZ", 941: "TINEJDAD", 647: "Tanger-Tétouan", 515: "AROUI",
                                                561: "TALIOUINE", 231: "RICH", 623: "OULED ABBOU", 177: "BIR JDID", 619: "SIDI HAJJAJ",
                                                476: "OURIKA", 826: "M'RIRT", 940: "SOUK EL SEBT", 827: "MECHRA BEL KSIRI", 938: "ASSILAH",
                                                939: "BOUIZAKARNE", 861: "EL BROUJ", 816: "SIDI ALLAL EL BAHRAOUI", 405: "Boujad", 942: "Sidi Rahal",
                                                943: "Tamaris", 944: "IMOUZZER"
                                            }
                                            data['Ville_Nom'] = data['Ville'].map(ville_mapping)

                                        # Calculer la fréquence des villes par cluster
                                        city_cluster_counts = data.groupby(['Cluster', 'Ville_Nom']).size().reset_index(name='Counts')

                                        # Créer le graphique Plotly
                                        fig = px.bar(city_cluster_counts, x='Ville_Nom', y='Counts', color='Cluster', barmode='group',
                                                    labels={'Ville_Nom': 'Nom de la Ville', 'Counts': 'Nombre de Cas', 'Cluster': 'Cluster'},
                                                    title='Répartition des Villes par Cluster')

                                        # Ajuster les labels des axes pour meilleure lisibilité
                                        fig.update_layout(xaxis_tickangle=-45)

                                        # Afficher le graphique
                                        st.plotly_chart(fig)

                                
                            # Déterminer si 'Sinistre' ou 'sinistre' est présent et afficher les prédictions
                            sinistre_col = None
                            if 'Sinistre' in data.columns:
                                sinistre_col = 'Sinistre'
                            elif 'sinistre' in data.columns:
                                sinistre_col = 'sinistre'
                            
                            if sinistre_col:
                                st.write("<div class='data-table'>Toutes les prédictions avec labels :</div>", unsafe_allow_html=True)
                                st.write(data[[sinistre_col, 'Cluster']])
                            else:
                                st.write("<div class='data-table'>Toutes les prédictions avec labels :</div>", unsafe_allow_html=True)
                                st.write(data[['Cluster']])
                                
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
