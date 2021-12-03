# Dependencies
import streamlit as st
import joblib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import requests
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Chargement des données initiales
@st.cache(allow_output_mutation=True)
def load_dataset_init():
    # Lecture des jeux de données
    train_init = pd.read_csv('application_train.csv')
    test_init = pd.read_csv('application_test.csv')

    return train_init, test_init


# Chargement des données normalisées
@st.cache(allow_output_mutation=True)
def load_dataset_norm():
    train_norm_init = pd.read_csv('P7_train_norm.csv', sep='\t')
    test_norm_init = pd.read_csv('P7_test_norm.csv', sep='\t')

    return train_norm_init, test_norm_init


# Chargement du modèle pour shap
@st.cache(allow_output_mutation=True)
def load_model_local():
    local_model = joblib.load('final_model.pkl')

    return local_model


@st.cache(allow_output_mutation=True)
def get_model_predictions(input):
    #mdl_url = 'http://127.0.0.1:5000/predict'
    mdl_url = 'http://francoischaumet.pythonanywhere.com/predict'
    data_json = {'data': input}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    #headers = {'Content-type': 'application/json'}
    prediction = requests.post(mdl_url, json=data_json, headers=headers)
    # predicted = json.loads(prediction.content)
    predicted = json.loads(prediction.content.decode("utf-8"))

    return predicted


@st.cache(allow_output_mutation=True)
def load_id(dataframe):
    ids = dataframe['SK_ID_CURR'].sort_values().unique()

    return ids


def main():
    # Définition du titre de l'app
    st.title('Credit Default Risk')
    st.markdown('---')

    with st.sidebar.container():
        page = st.sidebar.selectbox('Page navigation', ['Prediction', 'Data Analyse'])
        st.markdown('---')

    # Cas de la prédiction
    if page == 'Prediction':

        with st.spinner('Chargement des données'):
            train_norm_init, test_norm_init = load_dataset_norm()
        st.success('Données chargées, et disponibles !')  # todo Supprimer le message après un certain temps

        # On crée une copie de ces datasets pour ne pas les altérer
        train_norm = train_norm_init.copy()
        test_norm = test_norm_init.copy()

        sk_id = load_id(train_norm)

        selected_id = st.sidebar.selectbox("Veuillez sélectionner l'ID du customer", sk_id.astype('int32'))

        st.markdown('Utilisation en local')
        predict_btn = st.button('Prédire')
        if predict_btn: # todo Afficher les infos du client sous forme de table
            cli_data = train_norm[train_norm['SK_ID_CURR'] == selected_id]
            pred = cli_data['TARGET'].values
            #st.markdown('<font color=green>Le client est solvable {:,.2f}%</font>'.format(pred[0, 0] * 100),
            #            unsafe_allow_html=True)
            #st.markdown('<font color=red>Celui-ci est donc non solvable à {:,.2f}%</font>'.format(pred[0, 1] * 100),
            #            unsafe_allow_html=True)
            if pred == 0:
                st.markdown('<font color=green>Le client est solvable</font>', unsafe_allow_html=True)
                st.table(cli_data)
            elif pred == 1:
                st.markdown('<font color=red>Le client est en défaut de paiement</font>', unsafe_allow_html=True)
                st.table(cli_data)


        st.markdown('Requête via API')
        proba_btn = st.button('Probabilité')
        if proba_btn:
            # todo En cours de test
            cli_json = json.loads(train_norm[train_norm['SK_ID_CURR'] == selected_id].drop(columns=['SK_ID_CURR', 'TARGET']).to_json(orient='records'))[0]
            pred = get_model_predictions(cli_json)
            if pred['Prediction'] == [0]:
                st.markdown('<font color=green>Le client est solvable</font>', unsafe_allow_html=True)
            elif pred['Prediction'] == [1]:
                st.markdown('<font color=red>Le client est en défaut de paiement</font>', unsafe_allow_html=True)
            else:
                st.write('Ne reconnait pas la comparaison: vérifier le code')



    # Cas de la data analyse:
    if page == 'Data Analyse':

        with st.spinner('Chargement des données'):
            train_init, test_init = load_dataset_init()
        st.success('Données chargées, et disponibles !')  # todo Supprimer le message après un certain temps

        # On crée une copie de ces datasets pour ne pas les altérer
        train_data = train_init.copy()
        test_data = test_init.copy()

        # Création des plots
        def plot_vars(var1, var2):
            if train_data[var1].dtypes != 'object' and train_data[var2].dtypes != 'object':
                with st.echo():
                    graph = px.scatter(data_frame=train_data, x=var1, y=var2, color='TARGET')
            elif train_data[var1].dtypes == 'object' and train_data[var2].dtypes != 'object':
                with st.echo():
                    graph = px.violin(data_frame=train_data, x=var1, y=var2, color='TARGET')
            elif train_data[var1].dtypes != 'object' and train_data[var2].dtypes == 'object':
                with st.echo():
                    graph = px.violin(data_frame=train_data, x=var1, y=var2, color='TARGET')
            else:
                with st.echo():
                    graph = px.scatter(data_frame=train_data, x=var1, y=var2, color='TARGET')

            return graph

        # On va récupérer les colonnes dans une liste
        cols_to_plot = train_data.columns.tolist()

        # On supprime les colonnes à ne pas représenter
        cols_to_plot.remove('SK_ID_CURR')
        cols_to_plot.remove('TARGET')

        # Création des inputs
        selected_var1 = "Veuillez choisir une valeur"
        selected_var2 = "Veuillez choisir une valeur"

        # Définition de l'input 1
        if selected_var1 == "Veuillez choisir une valeur":
            selected_var1 = st.sidebar.selectbox("Première variable à explorer", cols_to_plot)

        # Définition de l'input 2
        if selected_var2 == "Veuillez choisir une valeur": # todo Supprimer le choix 1
            # On supprime la première variable sélectionnée de la liste
            cols_left = cols_to_plot.copy()
            #cols_left = cols_left.remove(selected_var1)
            selected_var2 = st.sidebar.selectbox("Deuxième variable à explorer", cols_left)


        # Représentation
        plot_button = st.button("Graphique bi-varié")
        if plot_button:
            fig = plt.figure()
            ax = plot_vars(selected_var1, selected_var2)
            st.plotly_chart(ax, use_container_width=True)

        # Heatmap des corrélations
        corr_button = st.button("Graphique de corrélation")
        if corr_button:
            df_corr = train_data.corr()
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(x=df_corr.columns, y=df_corr.index, z=np.array(df_corr))
            )
            fig.update_layout(title="Graphique de corrélation")
            st.plotly_chart(fig)

        # Réduction de dimension PCA # todo A modifier, jeu de données à retraiter
        pca_button = st.button("Réduction de dimension PCA")
        if pca_button:
            df_reduc = pd.get_dummies(train_data)
            scaler = StandardScaler()
            df_pca = scaler.fit_transform(df_reduc)
            pca = PCA(n_components=2)
            components = pca.fit_transform(df_pca)
            fig = px.scatter(components, color=train_data['TARGET'],
                             title="Analyse en Composantes Principales")
            st.plotly_chart(fig)

        # Réduction de dimension tSNE
        tsne_button = st.button("Réduction de dimension")
        if tsne_button:
            st.write("Non fonctionnel pour le moment")

if __name__ == '__main__':
    main()
