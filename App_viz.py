# Dependencies
import streamlit as st
import joblib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap

import requests
import json


# Chargement des données initiales
@st.cache(allow_output_mutation=True)
def load_dataset_init():
    train_init = pd.read_csv('train_poc.csv', sep='\t')
    test_init = pd.read_csv('test_poc.csv', sep='\t')

    return train_init, test_init


# Chargement des données normalisées
@st.cache(allow_output_mutation=True)
def load_dataset_norm():
    train_norm_init = pd.read_csv('P7_train_norm.csv', sep='\t')
    test_norm_init = pd.read_csv('P7_test_norm.csv', sep='\t')

    return train_norm_init, test_norm_init


# Si besoin de lire des fichiers zippés, utiliser ce code
# import zipfile
# zf_a = zipfile.ZipFile('P7_train_norm.zip')
# zf_b = zipfile.ZipFile('P7_test_norm.zip')
# @st.cache(allow_output_mutation=True)
# def load_dataset_norm():
#    train_norm_init = pd.read_csv(zf_a.open('P7_train_norm.csv'))
#    test_norm_init = pd.read_csv(zf_b.open('P7_test_norm.csv'))

#    return train_norm_init, test_norm_init


# Chargement du modèle pour shap
@st.cache(allow_output_mutation=True)
def load_model_local():
    local_model = joblib.load('best_model.pkl')

    return local_model


@st.cache(allow_output_mutation=True)
def get_model_predictions(input):
    # Ligne ci dessous pour test en local
    # mdl_url = 'http://127.0.0.1:5000/predict'
    # Ligne ci dessous pour test en ligne
    mdl_url = 'http://francoischaumet.pythonanywhere.com/predict'
    data_json = {'data': input}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    prediction = requests.post(mdl_url, json=data_json, headers=headers)
    predicted = json.loads(prediction.content.decode("utf-8"))

    return predicted


@st.cache(allow_output_mutation=True)
def load_id(dataframe):
    ids = dataframe['SK_ID_CURR'].sort_values().unique()

    return ids


@st.cache(allow_output_mutation=True)
def load_explainer(mdl):
    explainer = shap.TreeExplainer(mdl)

    return explainer


def main():
    st.title('Credit Default Risk - online app')
    st.markdown('---')

    with st.sidebar.container():
        page = st.sidebar.selectbox('Page navigation', ['Veuillez choisir:',
                                                        'Prediction',
                                                        'Data Analyse'])
        st.markdown('---')

    # Page d'accueil
    if page == 'Veuillez choisir:':
        st.subheader('Bienvenue sur ce modèle de prédiction en ligne')
        st.write('')
        st.warning('Merci de sélectionner un module dans la liste déroulante de gauche')

    # Cas de la prédiction
    if page == 'Prediction':
        st.subheader('Module de prédiction')
        st.write('')

        with st.spinner('Chargement des données'):
            train_norm_init, test_norm_init = load_dataset_norm()
            local_model = load_model_local()
            explainer = load_explainer(local_model)
        st.success('Données, modèle et explainer chargés, et disponibles !')

        # On crée une copie de ces datasets pour ne pas les altérer
        train_norm = train_norm_init.copy()
        test_norm = test_norm_init.copy()

        sk_id_train = load_id(train_norm)
        sk_id_test = load_id(test_norm)

        selected_id_pred = st.sidebar.selectbox('Veuillez sélectionner l\'ID du customer, pour la prédiction:',
                                                sk_id_train.astype('int32'))
        selected_id_prob = st.sidebar.selectbox('Veuillez sélectionner l\'ID du customer, pour la probabilité:',
                                                sk_id_test.astype('int32'))

        st.write('')
        st.warning('Veuillez cliquer pour afficher les résultats de la prédiction:')
        predict_btn = st.button('Prédire')
        if predict_btn:
            # On récupère les résultats via l'API
            cli_json = json.loads(
                train_norm[train_norm['SK_ID_CURR'] == selected_id_pred].drop(columns=['SK_ID_CURR', 'TARGET']).to_json(
                    orient='records'))[0]
            results_api = get_model_predictions(cli_json)

            # On récupère la valeur réelle
            cli_data = train_norm[train_norm['SK_ID_CURR'] == selected_id_pred]
            pred = cli_data['TARGET']

            if results_api['Prediction'][0] == 0:
                st.markdown('<font color=green>Le client est solvable</font>', unsafe_allow_html=True)
            elif results_api['Prediction'][0] == 1:
                st.markdown('<font color=red>Le client est en défaut de paiement</font>', unsafe_allow_html=True)

            # On compare les résultats
            col1, col2 = st.columns(2)
            col1.write('Classe prédite:')
            col1.write(pred)
            col2.write('Classe réelle:')
            col2.write(results_api['Prediction'][0])

            # On affiche les informations du client
            st.write('Données (normalisées) du client:')
            st.write(cli_data)
            st.write('')

            # Analyse des résultats avec l'explainer
            st.info('Facteurs les plus influents dans ce résultat:')
            exp_data = cli_data.drop(columns=['SK_ID_CURR', 'TARGET'])
            exp = explainer.shap_values(exp_data)
            # Force plot
            force_plt = shap.force_plot(np.round(explainer.expected_value[1], 3),
                                        np.round(exp[1], 3),
                                        np.round(exp_data, 3),
                                        matplotlib=True,
                                        show=False)
            st.pyplot(force_plt)

        st.write('')
        st.warning('Veuillez cliquer pour afficher les résultats de la probabilité (via API):')
        proba_btn = st.button('Probabilité')
        if proba_btn:
            # On récupère les résultats via l'API
            cli_json = json.loads(
                test_norm[test_norm['SK_ID_CURR'] == selected_id_prob].drop(columns=['SK_ID_CURR']).to_json(
                    orient='records'))[0]
            results_api = get_model_predictions(cli_json)

            # On affiche en fonction de la probabilité
            if results_api['Probabilite'][0][1] < 0.20:
                st.markdown('<font color=green>Faible probabilité d\'échec de remboursement.</font>',
                            unsafe_allow_html=True)
            elif results_api['Probabilite'][0][1] < 0.50:
                st.markdown('<font color=yellow>Risque d\'échec de remboursement, à contrôler.</font>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<font color=red>Forte probabilité d\'échec de remboursement, attention !</font>',
                            unsafe_allow_html=True)

            # Proba classe 0: prêt remboursé à temps
            # st.write(results_api['Probabilite'][0][0])
            # Proba classe 1 : difficultés de remboursement
            # st.write(results_api['Probabilite'][0][1])

            # Analyse des résultats avec l'explainer
            with st.spinner('Calcul de l\'explication en cours'):
                exp = explainer(test_norm.drop(columns=['SK_ID_CURR']))

            st.info('Représentation des features qui ont amené à ce résultat:')

            with st.spinner('Graphique en cours de préparation...'):
                # Attention au warning disabled suivant, à vérifier de temps en temps
                st.set_option('deprecation.showPyplotGlobalUse', False)
                # Création du graphique
                sum_plt = shap.summary_plot(shap_values=np.take(exp.values, 0, axis=-1),
                                            features=test_norm.drop(columns=['SK_ID_CURR']),
                                            feature_names=test_norm.drop(columns=['SK_ID_CURR']).columns,
                                            plot_size=(8, 8),
                                            sort=True,
                                            show=False)
                st.pyplot(sum_plt)

    # Cas de la data analyse:
    if page == 'Data Analyse':
        st.subheader("Module d'analyse")
        st.write('')

        with st.spinner('Chargement des données'):
            train_init, test_init = load_dataset_init()
        st.success('Données chargées, et disponibles !')

        # On crée une copie de ces datasets pour ne pas les altérer
        train_data = train_init.copy()
        # test_data = test_init.copy()

        st.warning('Veuillez sélectionner deux variables à explorer dans le menu de gauche')
        st.write('')

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

        selected_var1 = st.sidebar.selectbox('Première variable à explorer', cols_to_plot, index=5)

        # Suppression de la première feature sélectionnée
        cols_to_plot_b = cols_to_plot.copy()
        cols_to_plot_b.remove(selected_var1)

        selected_var2 = st.sidebar.selectbox('Deuxième variable à explorer', cols_to_plot_b)

        # Représentation
        st.write('')
        plot_button = st.button('Graphique bi-varié')
        if plot_button:
            graph_bi = plot_vars(selected_var1, selected_var2)
            st.plotly_chart(graph_bi, use_container_width=True)

        # Heatmap des corrélations
        st.write('')
        corr_button = st.button('Graphique de corrélation')
        if corr_button:
            df_corr = train_data[['TARGET', selected_var1, selected_var2]].corr()
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(x=df_corr.columns,
                           y=df_corr.index,
                           z=np.array(df_corr))
            )
            fig.update_layout(showlegend=True,
                              width=800,
                              height=600)
            st.plotly_chart(fig)


if __name__ == '__main__':
    main()
