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
# Création des sous-sections
with st.sidebar.expander("Analyse des Clusters", expanded=True):
    accueil = st.checkbox("Accueil")
    repartition_clusters = st.checkbox("Répartition des Clusters")
    repartition_villes = st.checkbox("Répartition des Villes par Cluster")
    repartition_propositions = st.checkbox("Répartition des Propositions par Cluster")
    types_propositions = st.checkbox("Répartition des Types de Proposition par Cluster")

with st.sidebar.expander("Montants", expanded=True):
    valeurs_boxplot = st.checkbox("Valeurs des Montants par Cluster en BoxPlot")
    valeurs_violin = st.checkbox("Valeurs des Montants par Cluster en Diagramme en Violin")
    pareto_montants = st.checkbox("Diagramme de Pareto pour Montants")
    montants_ville_frequent = st.checkbox("Montants par Ville la Plus Fréquente de Chaque Cluster")
    somme_journal = st.checkbox("Somme des Montants par Journal")
    moyenne_montants = st.checkbox("Moyenne des Montants par Cluster")
    nuage_points = st.checkbox("Diagramme de Nuage de Points pour Montants et Nombre de Propositions")
    tendances_annuelles = st.checkbox("Analyse des Tendances des Montants par Année")
    boxplot_types_proposition = st.checkbox("BoxPlot des Montants par Type de Proposition")

# Options sélectionnées
options = {
    "Accueil": accueil,
    "Répartition des Clusters": repartition_clusters,
    "Répartition des Villes par Cluster": repartition_villes,
    "Répartition des Propositions par Cluster": repartition_propositions,
    "Répartition des Types de Proposition par Cluster": types_propositions,
    "Valeurs des Montants par Cluster en BoxPlot": valeurs_boxplot,
    "Valeurs des Montants par Cluster en Diagramme en Violin": valeurs_violin,
    "Diagramme de Pareto pour Montants": pareto_montants,
    "Montants par Ville la Plus Fréquente de Chaque Cluster": montants_ville_frequent,
    "Somme des Montants par Journal": somme_journal,
    "Moyenne des Montants par Cluster": moyenne_montants,
    "Diagramme de Nuage de Points pour Montants et Nombre de Propositions": nuage_points,
    "Analyse des Tendances des Montants par Année": tendances_annuelles,
    "BoxPlot des Montants par Type de Proposition": boxplot_types_proposition
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
                                417: "TAN TAN", 421: "TANGER", 423: "TAOUNATE", 425: "TAOURIRT", 428: "TARFAYA",
                                432: "TAROUDANT", 435: "TAZA", 437: "TEMARA", 659: "TERRITOIRES SUD", 440: "TETOUAN",
                                442: "TIFELT", 702: "TIZNIT", 447: "YOUSSOUFIA", 897: "ZEUB"
                            }
                            
                            data['Ville_Nom'] = data['Ville'].map(ville_mapping)
                            
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
                                                    labels={'Cluster': 'Cluster', 'Count': 'Nombre de Sinistres'},
                                                    title='Répartition des Clusters')
                                        st.plotly_chart(fig)
                                        
                                        # Afficher la répartition des clusters avec labels
                                        st.write("Répartition des clusters avec labels :")
                                        cluster_distribution_labels = data.groupby('Cluster_Label').size().reset_index(name='Count')
                                        st.write(cluster_distribution_labels)
                                        
                                    elif option == "BoxPlot des Montants par Type de Proposition":
                                        st.subheader("BoxPlot des Montants par Type de Proposition")

                                        # Vérifier que les colonnes nécessaires sont présentes
                                        if 'Mnt' in data.columns and 'Type_pro' in data.columns:
                                            # Créer le BoxPlot
                                            fig_box = px.box(data, x='Type_pro', y='Mnt',
                                                            labels={'Type_pro': 'Type de Proposition', 'Mnt': 'Montant'},
                                                            title='BoxPlot des Montants par Type de Proposition')
                                            
                                            # Configurer le layout du graphique
                                            fig_box.update_layout(
                                                xaxis_title='Type de Proposition',
                                                yaxis_title='Montant',
                                                plot_bgcolor='rgba(240, 240, 240, 0.5)'
                                            )

                                            # Afficher le graphique
                                            st.plotly_chart(fig_box)
                                        else:
                                            st.error("Les colonnes nécessaires ('Mnt', 'Type_pro') ne sont pas présentes dans les données.")

                                        
                                    elif option == "Valeurs des Montants par Cluster en BoxPlot":
                                        st.subheader("BoxPlot")
                                        fig_box = px.box(data, y='Mnt', color='Cluster',
                                                        labels={'Mnt': 'Valeur du Montant', 'Cluster': 'Cluster'},
                                                        title='Diagramme en Boîte des Valeurs du Montant par Cluster')
                                        st.plotly_chart(fig_box)
                                        
                                    elif option == "Diagramme de Pareto pour Montants":
                                        st.subheader("Diagramme de Pareto pour Montants")
                                        
                                        # Nettoyer la colonne 'Mnt'
                                        data['Mnt'] = pd.to_numeric(data['Mnt'], errors='coerce')
                                        data['Mnt'].fillna(0, inplace=True)
                                        
                                        # Grouper les données par montant et compter les occurrences
                                        montant_counts = data['Mnt'].value_counts().reset_index()
                                        montant_counts.columns = ['Montant', 'Count']
                                        
                                        # Ordonner les montants par fréquence décroissante
                                        montant_counts = montant_counts.sort_values(by='Montant', ascending=False).reset_index(drop=True)
                                        
                                        # Calculer la somme cumulative des montants et des fréquences
                                        montant_counts['Cumulative Count'] = montant_counts['Count'].cumsum()
                                        montant_counts['Cumulative Percentage'] = 100 * montant_counts['Cumulative Count'] / montant_counts['Count'].sum()
                                        
                                        # Créer le diagramme de Pareto
                                        fig_pareto = go.Figure()

                                        # Ajouter les barres pour les montants
                                        fig_pareto.add_trace(go.Bar(
                                            x=montant_counts['Montant'],
                                            y=montant_counts['Count'],
                                            name='Fréquence',
                                            marker=dict(color='rgba(55, 83, 109, 0.7)')
                                        ))

                                        # Ajouter la ligne cumulative
                                        fig_pareto.add_trace(go.Scatter(
                                            x=montant_counts['Montant'],
                                            y=montant_counts['Cumulative Percentage'],
                                            mode='lines+markers',
                                            name='Cumulative Percentage',
                                            yaxis='y2',
                                            line=dict(color='rgba(219, 64, 82, 0.8)')
                                        ))

                                        # Configurer le layout du graphique
                                        fig_pareto.update_layout(
                                            title='Diagramme de Pareto pour Montants',
                                            xaxis_title='Montant',
                                            yaxis_title='Fréquence',
                                            yaxis2=dict(
                                                title='Pourcentage Cumulatif',
                                                titlefont=dict(color='rgba(219, 64, 82, 0.8)'),
                                                tickfont=dict(color='rgba(219, 64, 82, 0.8)'),
                                                overlaying='y',
                                                side='right'
                                            ),
                                            barmode='group'
                                        )

                                        # Afficher le graphique
                                        st.plotly_chart(fig_pareto)

                                        
                                    elif option == "Analyse des Tendances des Montants par Année":
                                        st.subheader("Analyse des Tendances des Montants par Année")
                                        
                                        # Nettoyer la colonne 'Mnt' et 'annee'
                                        data['Mnt'] = pd.to_numeric(data['Mnt'], errors='coerce')
                                        data['Mnt'].fillna(0, inplace=True)
                                        
                                        # Convertir la colonne 'annee' en numérique si ce n'est pas déjà fait
                                        data['annee'] = pd.to_numeric(data['annee'], errors='coerce')
                                        
                                        # Grouper les données par année et calculer la somme des montants pour chaque année
                                        trend_data = data.groupby('annee')['Mnt'].sum().reset_index()
                                        
                                        # Créer le graphique linéaire
                                        fig_trend = go.Figure()

                                        # Ajouter la ligne des tendances
                                        fig_trend.add_trace(go.Scatter(
                                            x=trend_data['annee'],
                                            y=trend_data['Mnt'],
                                            mode='lines+markers',
                                            name='Montants',
                                            line=dict(color='rgba(55, 83, 109, 0.8)'),
                                            marker=dict(color='rgba(55, 83, 109, 0.7)')
                                        ))

                                        # Configurer le layout du graphique
                                        fig_trend.update_layout(
                                            title='Tendances des Montants par Année',
                                            xaxis_title='Année',
                                            yaxis_title='Montant Total',
                                            xaxis=dict(
                                                tickmode='linear',  # Assure que toutes les années sont affichées
                                                dtick=1  # Intervalle des années (1 an)
                                            ),
                                            yaxis=dict(
                                                title='Montant Total'
                                            ),
                                            plot_bgcolor='rgba(240, 240, 240, 0.5)'
                                        )

                                        # Afficher le graphique
                                        st.plotly_chart(fig_trend)

                                        
                                    elif option == "Valeurs des Montants par Cluster en Diagramme en Violin":
                                        st.subheader("Diagramme en Violin")
                                        fig_violin = px.violin(data, y='Mnt', color='Cluster',
                                                            labels={'Mnt': 'Valeur du Montant', 'Cluster': 'Cluster'},
                                                            title='Diagramme en Violin des Valeurs du Montant par Cluster')
                                        st.plotly_chart(fig_violin)
                                        
                                    elif option == "Répartition des Villes par Cluster":
                                        st.subheader("Histogramme des Villes par Cluster")
                                        
                                        # Grouper par 'Cluster' et 'Ville' et compter les occurrences
                                        ville_cluster = data.groupby(['Cluster', 'Ville_Nom']).size().reset_index(name='Count')
                                        
                                        # Initialiser un DataFrame pour stocker les résultats finaux
                                        villes_finales = pd.DataFrame(columns=['Cluster', 'Ville_Nom', 'Count'])
                                        villes_utilisees = set()  # Pour garder une trace des villes déjà utilisées
                                        
                                        # Boucle pour chaque cluster
                                        for cluster in ville_cluster['Cluster'].unique():
                                            cluster_data = ville_cluster[ville_cluster['Cluster'] == cluster].sort_values(by='Count', ascending=False).reset_index(drop=True)
                                            
                                            # Trouver la ville la plus fréquente qui n'a pas été utilisée dans les clusters précédents
                                            for idx, row in cluster_data.iterrows():
                                                if row['Ville_Nom'] not in villes_utilisees:
                                                    villes_finales = pd.concat([villes_finales, pd.DataFrame([row])], ignore_index=True)
                                                    villes_utilisees.add(row['Ville_Nom'])
                                                    break
                                        
                                        # Afficher le DataFrame des villes les plus fréquentes par cluster
                                        st.write("<div class='data-table'>Villes les plus fréquentes par Cluster :</div>", unsafe_allow_html=True)
                                        st.write(villes_finales)
                                        
                                        # Créer et afficher le graphique
                                        fig_ville_cluster = go.Figure()

                                        # Ajouter les barres pour les villes les plus fréquentes
                                        fig_ville_cluster.add_trace(go.Bar(
                                            x=villes_finales['Cluster'],
                                            y=villes_finales['Count'],
                                            marker=dict(color='rgba(55, 83, 109, 0.7)'),
                                            name='Ville la plus fréquente'
                                        ))

                                        # Configurer le layout du graphique
                                        fig_ville_cluster.update_layout(
                                            title='Histogramme des Villes les Plus Fréquentes par Cluster',
                                            xaxis_title='Cluster',
                                            yaxis_title='Nombre de villes',
                                            barmode='group'
                                        )

                                        # Afficher le graphique
                                        st.plotly_chart(fig_ville_cluster)

                                    elif option == "Somme des Montants par Journal":
                                        st.subheader("Somme des Montants par Journal")
                                        if 'Mnt' in data.columns and 'Jnl' in data.columns:
                                            # Nettoyer la colonne 'Mnt'
                                            data['Mnt'] = pd.to_numeric(data['Mnt'], errors='coerce')
                                            data['Mnt'].fillna(0, inplace=True)
                                            
                                            # Préparer les données pour le graphique
                                            somme_montants = data.groupby('Jnl')['Mnt'].sum().reset_index()
                                            
                                            # Créer le graphique
                                            fig = px.bar(somme_montants, x='Jnl', y='Mnt',
                                                        title='Somme des Montants par Journal',
                                                        labels={'Jnl': 'Journal', 'Mnt': 'Somme des Montants'},
                                                        color='Mnt')
                                            st.plotly_chart(fig)
                                        else:
                                            st.error("Les colonnes nécessaires ('Mnt', 'Jnl') ne sont pas présentes dans les données.")
                                    
                                    elif option == "Montants par Ville la Plus Fréquente de Chaque Cluster":
                                        st.subheader("Histogramme des Montants par Ville la Plus Fréquente de Chaque Cluster")
                                        
                                        # Grouper par 'Cluster' et 'Ville' et compter les occurrences
                                        ville_cluster = data.groupby(['Cluster', 'Ville_Nom']).size().reset_index(name='Count')
                                        
                                        # Initialiser un DataFrame pour stocker les résultats finaux
                                        villes_finales = pd.DataFrame(columns=['Cluster', 'Ville_Nom', 'Count'])
                                        villes_utilisees = set()  # Pour garder une trace des villes déjà utilisées
                                        
                                        # Boucle pour chaque cluster
                                        for cluster in ville_cluster['Cluster'].unique():
                                            cluster_data = ville_cluster[ville_cluster['Cluster'] == cluster].sort_values(by='Count', ascending=False).reset_index(drop=True)
                                            
                                            # Trouver la ville la plus fréquente qui n'a pas été utilisée dans les clusters précédents
                                            for idx, row in cluster_data.iterrows():
                                                if row['Ville_Nom'] not in villes_utilisees:
                                                    villes_finales = pd.concat([villes_finales, pd.DataFrame([row])], ignore_index=True)
                                                    villes_utilisees.add(row['Ville_Nom'])
                                                    break

                                        # Filtrer les montants pour les villes les plus fréquentes
                                        villes_finales = villes_finales.rename(columns={'Ville_Nom': 'Ville'})
                                        filtered_data = data[data['Ville_Nom'].isin(villes_finales['Ville'])]

                                        # Créer un DataFrame pour les montants par ville et cluster
                                        montants_villes = filtered_data.groupby(['Cluster', 'Ville_Nom'])['Mnt'].sum().reset_index()
                                        
                                        # Créer l'histogramme
                                        fig_histogramme = px.histogram(filtered_data, x='Mnt', color='Ville_Nom',
                                                                    title='Distribution des Montants par Ville la Plus Fréquente de Chaque Cluster',
                                                                    labels={'Mnt': 'Montant', 'Ville_Nom': 'Ville'},
                                                                    histfunc='count')

                                        # Afficher le graphique
                                        st.plotly_chart(fig_histogramme)

                                    
                                    elif option == "Moyenne des Montants par Cluster":
                                        st.subheader("Moyenne des Montants par Cluster")
                                        moyenne_montants = data.groupby('Cluster')['Mnt'].mean().reset_index()
                                        fig_moyenne_montants = px.bar(moyenne_montants, x='Cluster', y='Mnt',
                                                                    labels={'Cluster': 'Cluster', 'Mnt': 'Moyenne des Montants'},
                                                                    title='Moyenne des Montants par Cluster')
                                        st.plotly_chart(fig_moyenne_montants)
                                    
                                    elif option == "Répartition des Propositions par Cluster":
                                        st.subheader("Répartition des Propositions par Cluster")
                                        propositions_cluster = data.groupby('Cluster')['Nb_propositions'].sum().reset_index()
                                        fig_propositions_cluster = px.bar(propositions_cluster, x='Cluster', y='Nb_propositions',
                                                                        labels={'Cluster': 'Cluster', 'Nb_propositions': 'Répartition des Propositions'},
                                                                        title='Répartition des Propositions par Cluster')
                                        st.plotly_chart(fig_propositions_cluster)
                                    
                                    elif option == "Répartition des Types de Proposition par Cluster":
                                        st.subheader("Répartition des Types de Proposition par Cluster")
                                        type_pro_cluster = data.groupby(['Cluster', 'Type_pro']).size().reset_index(name='Count')
                                        fig_type_pro_cluster = px.bar(type_pro_cluster, x='Cluster', y='Count', color='Type_pro',
                                                                    labels={'Cluster': 'Cluster', 'Count': 'Nombre de Propositions', 'Type_pro': 'Type de Proposition'},
                                                                    title='Répartition des Types de Proposition par Cluster')
                                        st.plotly_chart(fig_type_pro_cluster)
                                    
                                    elif option == "Diagramme de Nuage de Points pour Montants et Nombre de Propositions":
                                        st.subheader("Diagramme de Nuage de Points pour Montants et Nombre de Propositions")
                                        fig_scatter = px.scatter(data, x='Nb_propositions', y='Mnt', color='Cluster',
                                                                labels={'Nb_propositions': 'Nombre de Propositions', 'Mnt': 'Valeur du Montant'},
                                                                title='Diagramme de Nuage de Points pour Montants et Nombre de Propositions')
                                        st.plotly_chart(fig_scatter)
                                    
                                   


                                        
                                    

                                       
                                    
                                    

                                    


                                        
                                        
                                        
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
                            st.error(f"Erreur lors de la prédiction des clusters: {e}")
                    
                except Exception as e:
                    st.error(f"Erreur lors du prétraitement des données: {e}")
        except Exception as e:
            st.error(f"Erreur lors de la conversion des colonnes: {e}")
    else:
        st.error("Le fichier CSV ne contient pas les colonnes attendues.")
else:
    st.write("Veuillez télécharger un fichier CSV pour commencer.")