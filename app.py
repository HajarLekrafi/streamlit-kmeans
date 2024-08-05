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
    st.write("Données chargées :", unsafe_allow_html=True)
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
                                        st.markdown("<h2 style='color: #197d9f;'>Répartition des Clusters</h2>", unsafe_allow_html=True)

                                        cluster_distribution = data['Cluster'].value_counts().reset_index()
                                        cluster_distribution.columns = ['Cluster', 'Count']
                                        cluster_distribution['Label'] = cluster_distribution['Cluster'].map(labels)

                                        # Afficher le graphique
                                        fig = px.bar(cluster_distribution, x='Cluster', y='Count',
                                                    labels={'Cluster': 'Cluster', 'Count': 'Nombre de Sinistres'},
                                                    )
                                        st.plotly_chart(fig)

                                        # Afficher le tableau des labels
                                        st.markdown("<h3 style='color: #197d9f;'>Détails des Clusters</h3>", unsafe_allow_html=True)
                                        st.table(cluster_distribution[['Cluster', 'Label', 'Count']])

                                        # Analyse
                                        total_sinistres = cluster_distribution['Count'].sum()
                                        cluster_max = cluster_distribution.loc[cluster_distribution['Count'].idxmax()]
                                
                                        st.markdown(f"""
                                            <div class="features">
                                                <div class="feature">
                                                    <h2>Analyse</h2>
                                                    <p>La répartition des sinistres parmi les clusters montre la fréquence relative de chaque cluster. Cluster {cluster_max['Cluster']} a le plus grand nombre de sinistres, représentant 
                                                {cluster_max['Count'] / total_sinistres:.1%} du total des sinistres.</p>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)


                                    elif option == "Valeurs des Montants par Cluster en Diagramme en Violin":
                                        st.markdown("<h2 style='color: #197d9f;'>Diagramme en Violin des Valeurs du Montant par Cluster</h2>", unsafe_allow_html=True)


                                         # Créer le graphique
                                        fig_violin = px.violin(data, y='Mnt', color='Cluster',
                                                        labels={'Mnt': 'Valeur du Montant', 'Cluster': 'Cluster'}
                                                        )
                                        st.plotly_chart(fig_violin)

                                        # Dictionnaire pour les labels des clusters
                                        cluster_labels = {
                                            0: 'Cluster 0 ',
                                            1: 'Cluster 1 ',
                                            2: 'Cluster 2 ',
                                            3: 'Cluster 3 '
                                            # Ajoutez d'autres clusters et labels si nécessaire
                                        }
                                        
                                        st.write(f"- **Distribution Globale** : Le diagramme en violon montre la distribution des montants dans ce cluster. Un violon plus large indique une concentration élevée des montants à certaines valeurs, tandis qu'un violon plus étroit montre une concentration plus faible.")
                                        st.write(f"- **Comparaison entre les Clusters** :Un cluster avec un violon plus large à une hauteur spécifique peut avoir des montants plus courants à cette valeur, tandis que des violons plus étroits peuvent indiquer des montants moins fréquents.")

                                        # Préparez les données pour chaque cluster
                                        for cluster in data['Cluster'].unique():
                                            subset = data[data['Cluster'] == cluster]
                                            median_mnt = subset['Mnt'].median()
                                            q1 = subset['Mnt'].quantile(0.25)
                                            q3 = subset['Mnt'].quantile(0.75)
                                            min_mnt = subset['Mnt'].min()
                                            max_mnt = subset['Mnt'].max()
                                            label = cluster_labels.get(cluster, f'Cluster {cluster}')
                                            
                                            # Préparez le texte dynamique
                                            analyse_text = (
                                                f"<strong>Analyse pour {label} :</strong><br>"
                                                f"<strong>- Médiane : </strong>La médiane est de {median_mnt:.2f}. Cette valeur sépare les montants en deux groupes égaux, avec la moitié des montants au-dessus et l'autre moitié en dessous.<br>"
                                                f"<strong>- Écart Interquartile (IQR) : </strong>La largeur entre Q1 ({q1:.2f}) et Q3 ({q3:.2f}) montre où se situe la majorité des montants. Un large écart indique une grande variation des montants, tandis qu'un écart étroit suggère une plus grande homogénéité.<br>"
                                                f"<strong>- Montants Extrêmes : </strong>Les valeurs minimales ({min_mnt:.2f}) et maximales ({max_mnt:.2f}) montrent l'étendue des montants dans ce cluster. Les valeurs extrêmes peuvent indiquer des cas atypiques ou des anomalies."
                                            )
                                            
                                            # Utilisez st.markdown pour inclure le texte dynamique dans le HTML
                                            st.markdown(f"""
                                            <div class="features">
                                                <div class="feature">
                                                    <p>{analyse_text}</p>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)


                                        
                                    elif option == "BoxPlot des Montants par Type de Proposition":
                                        st.markdown("<h2 style='color: #197d9f;'>BoxPlot des Montants par Type de Proposition</h2>", unsafe_allow_html=True)

                                        if 'Mnt' in data.columns and 'Type_pro' in data.columns:
                                            fig_box = px.box(data, x='Type_pro', y='Mnt',
                                                            labels={'Type_pro': 'Type de Proposition', 'Mnt': 'Montant'})
                                            fig_box.update_layout(
                                                xaxis_title='Type de Proposition',
                                                yaxis_title='Montant',
                                                plot_bgcolor='rgba(240, 240, 240, 0.5)'
                                            )
                                            st.plotly_chart(fig_box)
                                            
                                            # Dictionnaire pour les labels des types de proposition
                                            type_pro_labels = {
                                                'BJ': 'Procédure judiciaire',
                                                'TJ': 'Transaction suite jugement au fond',
                                                'TD': 'Transaction directe à l\'amiable'
                                            }

                                            # Analyse
                                            for type_pro in data['Type_pro'].unique():
                                                subset = data[data['Type_pro'] == type_pro]
                                                median_mnt = subset['Mnt'].median()
                                                label = type_pro_labels.get(type_pro, 'Type inconnu')
                                                st.write(f"**Analyse pour {label} :** Le montant médian des sinistres est de {median_mnt:.2f}. ")
                                            
                                            st.write("Les variations indiquent que les sinistres de ce type peuvent varier considérablement en montant, "
                                                    "ce qui pourrait suggérer une diversité dans les cas traités.")


                                        
                                    elif option == "Valeurs des Montants par Cluster en BoxPlot":
                                        st.markdown("<h2 style='color: #197d9f;'>BoxPlot des Valeurs du Montant par Cluster</h2>", unsafe_allow_html=True)


                                        # Créer le graphique
                                        fig_box = px.box(data, y='Mnt', color='Cluster',
                                                        labels={'Mnt': 'Valeur du Montant', 'Cluster': 'Cluster'}
                                                        )
                                        st.plotly_chart(fig_box)

                                        # Dictionnaire pour les labels des clusters
                                        cluster_labels = {
                                            0: 'Cluster 0 ',
                                            1: 'Cluster 1 ',
                                            2: 'Cluster 2 ',
                                            3: 'Cluster 3 '
                                            # Ajoutez d'autres clusters et labels si nécessaire
                                        }
                                        
                                        # Analyse simplifiée
                                        for cluster in data['Cluster'].unique():
                                            subset = data[data['Cluster'] == cluster]
                                            median_mnt = subset['Mnt'].median()
                                            q1 = subset['Mnt'].quantile(0.25)
                                            q3 = subset['Mnt'].quantile(0.75)
                                            iqr = q3 - q1
                                            min_mnt = subset['Mnt'].min()
                                            max_mnt = subset['Mnt'].max()
                                            label = cluster_labels.get(cluster, f'Cluster {cluster}')
                                            
                                            
                                        st.markdown(f"""
                                            <div class="features">
                                                <div class="feature">
                                                    <p> Médiane: La médiane du montant est de {median_mnt:.2f}. Cela signifie que la moitié des propositions ont des montants inférieurs ou égaux à cette valeur.")
                                           Intervalle Interquartile (IQR)** : {iqr:.2f}. C'est la différence entre le premier et le troisième quartile, montrant combien les montants sont dispersés autour de la médiane.")
                                          Montants Minimum et Maximum** : Les montants varient de {min_mnt:.2f} à {max_mnt:.2f}, indiquant les valeurs les plus basses et les plus élevées dans ce groupe.")
                                            
                                        "En résumé, les boxplots montrent comment les montants des propositions sont distribués dans chaque groupe (cluster). "
                                                "Les différences entre les groupes peuvent indiquer des variations importantes dans les montants, ce qui peut nous aider à comprendre les caractéristiques des propositions dans chaque groupe et à repérer les valeurs extrêmes.")
</p>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                    elif option == "Analyse des Tendances des Montants par Année":
                                        st.markdown("<h2 style='color: #197d9f;'>Analyse des Tendances des Montants par Année</h2>", unsafe_allow_html=True)

                                        
                                        
                                        # Nettoyer la colonne 'Mnt' et 'annee'
                                        data['Mnt'] = pd.to_numeric(data['Mnt'], errors='coerce')
                                        data['Mnt'].fillna(0, inplace=True)
                                        
                                        # Convertir la colonne 'annee' en numérique si ce n'est pas déjà fait
                                        data['annee'] = pd.to_numeric(data['annee'], errors='coerce')
                                        
                                        # Grouper les données par année et calculer la somme des montants pour chaque année
                                        trend_data = data.groupby('annee')['Mnt'].sum().reset_index()
                                        
                                        # Créer le graphique linéaire
                                        fig_trend = go.Figure()
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
                                        
                                        # Analyse basée sur les données
                                        st.write("**Analyse des Tendances des Montants par Année :**")
                                        
                                        # Trouver les années avec les montants les plus élevés et les plus bas
                                        max_year = trend_data.loc[trend_data['Mnt'].idxmax()]
                                        min_year = trend_data.loc[trend_data['Mnt'].idxmin()]
                                        
                                        st.write(f"- **Année avec le Montant Total le Plus Élevé** : {max_year['annee']} avec un montant total de {max_year['Mnt']:.2f}. Cette année a enregistré le montant total le plus élevé, ce qui pourrait indiquer une augmentation des propositions ou des sinistres.")
                                        st.write(f"- **Année avec le Montant Total le Plus Bas** : {min_year['annee']} avec un montant total de {min_year['Mnt']:.2f}. Cette année a enregistré le montant total le plus bas, ce qui pourrait indiquer une diminution des propositions ou des sinistres.")
                                        st.write(f"- **Tendances Générales** : Le graphique linéaire montre comment les montants totaux évoluent d'année en année. Observez les augmentations ou les diminutions significatives. Par exemple, une tendance croissante pourrait indiquer des changements dans les politiques ou une augmentation des demandes.")
                                        st.write(f"- **Observations Particulières** : Notez les années avec des pics ou des creux marqués. Cela pourrait être dû à des événements spécifiques ou des changements dans les conditions économiques ou les pratiques de l'entreprise.")
    


                                        
                                    elif option == "Répartition des Villes par Cluster":
                                        st.markdown("<h2 style='color: #197d9f;'>Répartition des Villes par Cluster</h2>", unsafe_allow_html=True)
                                        ville_cluster = data.groupby(['Cluster', 'Ville_Nom']).size().reset_index(name='Count')
                                        villes_finales = pd.DataFrame(columns=['Cluster', 'Ville_Nom', 'Count'])
                                        villes_utilisees = set()
                                        for cluster in ville_cluster['Cluster'].unique():
                                            cluster_data = ville_cluster[ville_cluster['Cluster'] == cluster].sort_values(by='Count', ascending=False).reset_index(drop=True)
                                            for idx, row in cluster_data.iterrows():
                                                if row['Ville_Nom'] not in villes_utilisees:
                                                    villes_finales = pd.concat([villes_finales, pd.DataFrame([row])], ignore_index=True)
                                                    villes_utilisees.add(row['Ville_Nom'])
                                                    break
                                        
                                        
                                        
                                        # Analyse
                                        for cluster in villes_finales['Cluster'].unique():
                                            city_count = villes_finales[villes_finales['Cluster'] == cluster]
                                            most_common_city = city_count.loc[city_count['Count'].idxmax()]
                                            st.markdown(f"""
                                            <div class="features">
                                                <div class="feature">
                                                    <p><strong>Pour le Cluster {cluster} :</strong>
                                                    La ville la plus fréquente est <strong>{most_common_city['Ville_Nom']}</strong> avec {most_common_city['Count']} sinistres.</p>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        
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
                                            st.markdown("<h2 style='color: #197d9f;'>Somme des Montants par Journal</h2>", unsafe_allow_html=True)

                                            if 'Mnt' in data.columns and 'Jnl' in data.columns:
                                                # Nettoyer la colonne 'Mnt'
                                                data['Mnt'] = pd.to_numeric(data['Mnt'], errors='coerce')
                                                data['Mnt'].fillna(0, inplace=True)
                                                
                                                # Préparer les données pour le graphique
                                                somme_montants = data.groupby('Jnl')['Mnt'].sum().reset_index()
                                                
                                                # Créer le graphique
                                                fig = px.bar(somme_montants, x='Jnl', y='Mnt',
                                                            labels={'Jnl': 'Journal', 'Mnt': 'Somme des Montants'},
                                                            color='Mnt')
                                                st.plotly_chart(fig)
                                                
                                                # Analyse basée sur les données
                                                st.write("**Analyse des Sommes des Montants par Journal :**")
                                                
                                                # Trouver le journal avec le montant total le plus élevé et le plus bas
                                                max_journal = somme_montants.loc[somme_montants['Mnt'].idxmax()]
                                                min_journal = somme_montants.loc[somme_montants['Mnt'].idxmin()]
                                                
                                                st.write(f"- **Journal avec le Montant Total le Plus Élevé** : {max_journal['Jnl']} avec une somme de {max_journal['Mnt']:.2f}. Cela indique que ce journal a enregistré le montant total le plus élevé parmi tous les journaux.")
                                                st.write(f"- **Journal avec le Montant Total le Plus Bas** : {min_journal['Jnl']} avec une somme de {min_journal['Mnt']:.2f}. Ce journal a enregistré le montant total le plus bas.")
                                                st.write(f"- **Distribution des Montants** : L'histogramme montre comment les montants totaux sont répartis entre les différents journaux. Les journaux avec des barres plus longues indiquent une somme totale plus élevée, tandis que les barres plus courtes indiquent des montants totaux plus faibles.")

                                            else:
                                                st.error("Les colonnes nécessaires ('Mnt', 'Jnl') ne sont pas présentes dans les données.")

                                    
                                    elif option == "Montants par Ville la Plus Fréquente de Chaque Cluster":
                                            st.markdown("<h2 style='color: #197d9f;'>Histogramme des Montants par Ville la Plus Fréquente de Chaque Cluster</h2>", unsafe_allow_html=True)

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
                                                                            labels={'Mnt': 'Montant', 'Ville_Nom': 'Ville'},
                                                                            histfunc='count')

                                            # Afficher le graphique
                                            st.plotly_chart(fig_histogramme)

                                            st.write(f"- L'histogramme montre la répartition des montants pour ces villes. Les hauteurs des barres indiquent combien de propositions ont des montants dans les différentes plages de valeurs pour chaque ville.")
                                            st.write(f"- Une ville avec de nombreuses propositions à des montants élevés peut indiquer une concentration de propositions coûteuses dans cette ville.")
                                            st.write(f"- En comparant les villes entre les clusters, on peut voir quelles villes ont des montants plus élevés ou plus faibles et comment cela se compare aux autres clusters.")
                                            # Analyse basée sur les données
                                            for cluster in villes_finales['Cluster'].unique():
                                                villes_cluster = villes_finales[villes_finales['Cluster'] == cluster]['Ville'].values
                                                st.write(f"**Cluster {cluster} :**")
                                                st.write(f"- La ville la plus fréquente dans ce cluster est : {', '.join(villes_cluster)}.")
                                               



                                    
                                    elif option == "Moyenne des Montants par Cluster":
                                        st.markdown("<h2 style='color: #197d9f;'>Moyenne des Montants par Cluster</h2>", unsafe_allow_html=True)
                                    
                                        # Calculer la moyenne des montants par cluster
                                        moyenne_montants = data.groupby('Cluster')['Mnt'].mean().reset_index()
                                        
                                        # Créer le graphique
                                        fig_moyenne_montants = px.bar(moyenne_montants, x='Cluster', y='Mnt',
                                                                    labels={'Cluster': 'Cluster', 'Mnt': 'Moyenne des Montants'},
                                                                    )
                                        st.plotly_chart(fig_moyenne_montants)
                                        
                                        # Analyse basée sur les données
                                        st.write("**Analyse de la Moyenne des Montants par Cluster :**")
                                        
                                        # Trouver les clusters avec les moyennes les plus élevées et les plus basses
                                        max_cluster = moyenne_montants.loc[moyenne_montants['Mnt'].idxmax()]
                                        min_cluster = moyenne_montants.loc[moyenne_montants['Mnt'].idxmin()]
                                        
                                        st.write(f"- **Cluster avec la Moyenne des Montants la Plus Élevée** : Cluster {max_cluster['Cluster']} avec une moyenne de {max_cluster['Mnt']:.2f}. Ce cluster a les montants moyens les plus élevés, ce qui peut indiquer que les propositions dans ce cluster sont généralement plus coûteuses.")
                                        st.write(f"- **Cluster avec la Moyenne des Montants la Plus Basse** : Cluster {min_cluster['Cluster']} avec une moyenne de {min_cluster['Mnt']:.2f}. Ce cluster a les montants moyens les plus bas, ce qui peut suggérer que les propositions sont généralement moins coûteuses.")
                                        st.write(f"- **Distribution des Moyennes** : L'histogramme montre la moyenne des montants pour chaque cluster. Les hauteurs des barres indiquent les montants moyens dans chaque cluster, permettant de comparer directement les coûts moyens entre les clusters.")


                                    
                                    elif option == "Répartition des Propositions par Cluster":
                                        st.markdown("<h2 style='color: #197d9f;'>Répartition des Propositions par Cluster</h2>", unsafe_allow_html=True)
                                       
                                        # Calculer la répartition des propositions par cluster
                                        propositions_cluster = data.groupby('Cluster')['Nb_propositions'].sum().reset_index()
                                        
                                        # Créer le graphique
                                        fig_propositions_cluster = px.bar(propositions_cluster, x='Cluster', y='Nb_propositions',
                                                                        labels={'Cluster': 'Cluster', 'Nb_propositions': 'Répartition des Propositions'},
                                                                        )
                                        
                                        # Afficher le graphique
                                        st.plotly_chart(fig_propositions_cluster)
                                        
                                        # Analyse
                                        total_propositions = propositions_cluster['Nb_propositions'].sum()
                                        cluster_max_propositions = propositions_cluster.loc[propositions_cluster['Nb_propositions'].idxmax()]
                                        
                                        st.markdown(f"""
                                            <div class="features">
                                                <div class="feature">
                                                    <h2>Analyse des propositions par cluster </h2>
                                                    <p><La répartition des propositions parmi les clusters montre que le Cluster {cluster_max_propositions['Cluster']} possède le plus grand nombre de propositions, avec un total de {cluster_max_propositions['Nb_propositions']}. 
                                                Cela représente {cluster_max_propositions['Nb_propositions'] / total_propositions:.1%} du total des propositions. </p>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                                    elif option == "Répartition des Types de Proposition par Cluster":
                                        st.markdown("<h2 style='color: #197d9f;'>Répartition des Types de Proposition par Cluster</h2>", unsafe_allow_html=True)
                          
                                        # Calculer la répartition des types de proposition par cluster
                                        type_pro_cluster = data.groupby(['Cluster', 'Type_pro']).size().reset_index(name='Count')
                                        
                                        # Créer le graphique
                                        fig_type_pro_cluster = px.bar(type_pro_cluster, x='Cluster', y='Count', color='Type_pro',
                                                                    labels={'Cluster': 'Cluster', 'Count': 'Nombre de Propositions', 'Type_pro': 'Type de Proposition'},
                                                                    )
                                        
                                        # Afficher le graphique
                                        st.plotly_chart(fig_type_pro_cluster)
                                        
                                        # Dictionnaire pour les labels des types de proposition
                                        type_pro_labels = {
                                            'BJ': 'Procédure judiciaire',
                                            'TJ': 'Transaction suite jugement au fond',
                                            'TD': 'Transaction directe à l\'amiable'
                                        }
                                        
                                        # Analyse
                                        for cluster in type_pro_cluster['Cluster'].unique():
                                            cluster_data = type_pro_cluster[type_pro_cluster['Cluster'] == cluster]
                                            type_counts = cluster_data.groupby('Type_pro')['Count'].sum()
                                            most_common_type = type_counts.idxmax()
                                            most_common_type_label = type_pro_labels.get(most_common_type, 'Type inconnu')
                                            st.write(f"**Cluster {cluster} :** Le type de proposition le plus fréquent est '{most_common_type_label}' avec un total de {type_counts.max()} propositions. "
                                                    f"Ce type représente {type_counts.max() / type_counts.sum():.1%} du total des propositions dans ce cluster.")


                                    
                                    elif option == "Diagramme de Nuage de Points pour Montants et Nombre de Propositions":
                                        st.markdown("<h2 style='color: #197d9f;'>Diagramme de Nuage de Points pour Montants et Nombre de Propositions</h2>", unsafe_allow_html=True)

                                        # Créer le graphique de nuage de points
                                        fig_scatter = px.scatter(data, x='Nb_propositions', y='Mnt', color='Cluster',
                                                                labels={'Nb_propositions': 'Nombre de Propositions', 'Mnt': 'Valeur du Montant'})
                                        st.plotly_chart(fig_scatter)
                                        
                                        # Analyse basée sur les données
                                        st.write("**Analyse du Diagramme de Nuage de Points :**")
                                        
                                        # Trouver les tendances générales
                                        st.write("- **Tendances Générales :** Examinez la distribution des points pour comprendre la relation entre le nombre de propositions et la valeur du montant. Les points sont colorés par cluster, ce qui permet de voir comment chaque cluster se distribue en termes de montant et de nombre de propositions.")
                                        
                                        # Identifier les clusters avec des montants élevés et un grand nombre de propositions
                                        for cluster in data['Cluster'].unique():
                                            cluster_data = data[data['Cluster'] == cluster]
                                            if not cluster_data.empty:
                                                avg_nb_propositions = cluster_data['Nb_propositions'].mean()
                                                avg_montant = cluster_data['Mnt'].mean()
                                                st.write(f"- **Cluster {cluster} :**")
                                                st.write(f"  - Nombre moyen de propositions : {avg_nb_propositions:.2f}")
                                                st.write(f"  - Valeur moyenne du montant : {avg_montant:.2f}")                        
                                        
                                # Déterminer si 'Sinistre' ou 'sinistre' est présent et afficher les prédictions
                            sinistre_col = None
                            if 'Sinistre' in data.columns:
                                sinistre_col = 'Sinistre'
                            elif 'sinistre' in data.columns:
                                sinistre_col = 'sinistre'
                            
                            if sinistre_col:
                                st.write("Toutes les prédictions avec labels :", unsafe_allow_html=True)
                                st.write(data[[sinistre_col, 'Cluster']])
                            else:
                                st.write("Toutes les prédictions avec labels :", unsafe_allow_html=True)
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
    
# Custom CSS
st.markdown("""
    <style>
    .features {
        display: flex;
        justify-content: space-around;
        padding: 2rem 0;
        background: #fff;
    }
    .feature {
        background: #fff;
        padding: 1rem;
        margin: 1rem;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .feature h2 {
        margin-bottom: 1rem;
    }
    footer {
        background: #197d9f;
        color: #fff;
        text-align: center;
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)



# Features Section
st.markdown("""
<div class="features">
    <div class="feature">
        <h2>Pourquoi Clustering ? </h2>
        <p>Le clustering est crucial dans notre projet d'analyse des sinistres
        corporelles, car il permet de regrouper les données en ensembles homogènes,
        facilitant l'identification de schémas et de tendances.En optimisant les processus, le clustering réduit les coûts 
        et augmente l'efficacité opérationnelle, 
        renforçant ainsi la satisfaction des clients..</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<footer>
    <p>&copy; 2024 Cat Assurance et Réassurance</p>
</footer>
""", unsafe_allow_html=True)

    
