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

# Chargement du modèle en local pour shap
@st.cache(allow_output_mutation=True)
def load_model_local():
    local_model = joblib.load('best_model.pkl')

    return local_model


# Prédiction via API
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


# Chargement des identifiants pour un jeu de données passé en paramètres
@st.cache(allow_output_mutation=True)
def load_id(dataframe):
    ids = dataframe['SK_ID_CURR'].sort_values().unique()

    return ids


# Chargement de l'exlainer pour un modèle passé en paramètres
@st.cache(allow_output_mutation=True)
def load_explainer(mdl):
    explainer = shap.TreeExplainer(mdl)

    return explainer


# Application de l'explainer sur le dataframe, passés en paramètres
@st.cache(allow_output_mutation=True)
def run_explainer(xplnr, df):
    exp = xplnr(df.drop(columns=['SK_ID_CURR']))

    return exp


# Création des plots pour deux variables
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def plot_vars(df, var1, var2):
    if df[var1].dtypes != 'object' and df[var2].dtypes != 'object':
        with st.echo():
            graph = px.scatter(data_frame=df, x=var1, y=var2, color='TARGET')
    elif df[var1].dtypes == 'object' and df[var2].dtypes != 'object':
        with st.echo():
            graph = px.violin(data_frame=df, x=var1, y=var2, color='TARGET')
    elif df[var1].dtypes != 'object' and df[var2].dtypes == 'object':
        with st.echo():
            graph = px.violin(data_frame=df, x=var1, y=var2, color='TARGET')
    else:
        with st.echo():
            graph = px.scatter(data_frame=df, x=var1, y=var2, color='TARGET')

    return graph


def main():
    st.title('Application de prédiction du risque de défault de paiement d\'un client')
    st.markdown('---')
    st.write('')

    with st.sidebar.container():
        page = st.sidebar.selectbox('Page navigation', ['Veuillez choisir :',
                                                        'Prediction',
                                                        'Data Analyse'])

    # Page d'accueil
    if page == 'Veuillez choisir :':
        st.sidebar.write('---')
        st.subheader('Bienvenue sur ce modèle de prédiction en ligne.')
        st.subheader('Voici le détail des options disponibles :')
        st.write('')
        st.write('Module de prédiction :')
        st.write(' - *permet de prédire la solvabilité du client choisi*')
        st.write(' - *permet de déterminer la probabilité de remboursement du client choisi*')
        st.write('')
        st.write('Module de data-analyse :')
        st.write(' - *permet de représenter la relation entre deux variables sélectionnées*')
        st.write(' - *permet de représenter la corrélation entre deux variables sélectionnées et la cible*')
        st.write('')
        st.info('Pour commencer, merci de sélectionner le module à utiliser dans la liste déroulante de gauche.')

    # Cas de la prédiction
    if page == 'Prediction':
        st.sidebar.write('---')
        st.subheader('Module de prédiction')
        st.write('')
        st.write(' - *permet de prédire la solvabilité du client choisi*')
        st.write(' - *permet de déterminer la probabilité de remboursement du client choisi*')
        st.write('')
        st.caption('Pour rappel :')
        st.caption(' - *Classe 0 :* le prêt est remboursé dans les temps')
        st.caption(' - *Classe 1 :* le client a des difficultés de remboursement')
        st.write('')

        with st.spinner('Chargement des données'):
            train_norm_init, test_norm_init = load_dataset_norm()
            local_model = load_model_local()
            explainer = load_explainer(local_model)

        # On crée une copie de ces datasets pour ne pas les altérer
        train_norm = train_norm_init.copy()
        test_norm = test_norm_init.copy()

        # On charge les identifiants
        sk_id_train = load_id(train_norm)
        sk_id_test = load_id(test_norm)

        st.info('Pour commencer, veuillez sélectionner l\'ID du client dans la liste déroulante de gauche.')
        selected_id_pred = st.sidebar.selectbox('ID client pour la solvabilité :',
                                                sk_id_train.astype('int32'))
        st.sidebar.write('')
        selected_id_prob = st.sidebar.selectbox('ID client pour la probabilité de remboursement :',
                                                sk_id_test.astype('int32'))

        st.write('')
        cpred, cprob = st.columns(2)
        with cpred:
            st.write('Pour déterminer la solvabilité du client, veuillez cliquer ci-dessous :')
            predict_btn = st.button('Prédire')
        with cprob:
            st.write('Pour déterminer la probabilité de remboursement, veuillez cliquer ci-dessous :')
            proba_btn = st.button('Probabilité')

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
                st.write('')
                st.markdown('<font color=green>Ce client est prédit solvable</font>', unsafe_allow_html=True)
                st.write('')
            elif results_api['Prediction'][0] == 1:
                st.write('')
                st.markdown('<font color=red>Ce client est prédit en défaut de paiement</font>', unsafe_allow_html=True)
                st.write('')

            # On compare les résultats
            cls_pred1, cls_pred2, cls_real1, cls_real2 = st.columns(4)
            with cls_pred1, cls_pred2:
                cls_pred1.write('*Classe prédite :*')
                cls_pred2.markdown(results_api['Prediction'][0])
            with cls_real1, cls_real2:
                cls_real1.write('*Classe réelle :*')
                cls_real2.markdown(pred.values)

            # On affiche les informations du client
            st.write('')
            st.write('')
            st.write('Pour information, voici les données initiales (normalisées) du client :')
            st.write(cli_data)
            st.write('')

            # Analyse des résultats avec l'explainer
            st.write('')
            st.write('Les variables ayant eu le plus d\'influence dans ce résultat sont :')
            st.write('')
            exp_data = cli_data.drop(columns=['SK_ID_CURR', 'TARGET'])
            exp = explainer.shap_values(exp_data)
            # Force plot
            force_plt = shap.force_plot(np.round(explainer.expected_value[1], 3),
                                        np.round(exp[1], 3),
                                        np.round(exp_data, 3),
                                        matplotlib=True,
                                        show=False)
            st.pyplot(force_plt)

        if proba_btn:
            # On récupère les résultats via l'API
            cli_json = json.loads(
                test_norm[test_norm['SK_ID_CURR'] == selected_id_prob].drop(columns=['SK_ID_CURR']).to_json(
                    orient='records'))[0]
            results_api = get_model_predictions(cli_json)

            # On affiche en fonction de la probabilité
            c_prob1, c_prob2 = st.columns(2)
            st.write('')
            if results_api['Probabilite'][0][0] > 0.65:
                with c_prob1:
                    st.markdown('<font color=green>Forte probabilité de remboursement :</font>',
                                unsafe_allow_html=True)
                with c_prob2:
                    st.write("{:.2%}".format(results_api['Probabilite'][0][0]))
            elif results_api['Probabilite'][0][0] > 0.35:
                with c_prob1:
                    st.markdown('<font color=yellow>Risque d\'échec de remboursement, à surveiller :</font>',
                                unsafe_allow_html=True)
                with c_prob2:
                    st.write("{:.2%}".format(results_api['Probabilite'][0][0]))
            else:
                with c_prob1:
                    st.markdown('<font color=red>Forte probabilité d\'échec de remboursement, attention :</font>',
                                unsafe_allow_html=True)
                with c_prob2:
                    st.write("{:.2%}".format(results_api['Probabilite'][0][0]))

            # Proba classe 0: prêt remboursé à temps
            # st.write(results_api['Probabilite'][0][0])
            # Proba classe 1 : difficultés de remboursement
            # st.write(results_api['Probabilite'][0][1])

            # Analyse des résultats avec l'explainer
            exp = run_explainer(explainer, test_norm)

            st.write('')
            st.write('Représentation des variables qui ont amené à ce résultat :')
            st.write('')

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
        st.sidebar.write('---')
        st.subheader('Module d\'analyse')
        st.write('')
        st.write(' - *permet de représenter la relation entre les variables sélectionnées*')
        st.write(' - *permet de représenter la corrélation entre deux variables et la cible*')

        st.write('')

        with st.spinner('Chargement des données'):
            train_init, test_init = load_dataset_init()

        # On crée une copie de ces datasets pour ne pas les altérer
        train_data = train_init.copy()
        # test_data = test_init.copy()

        st.info('Veuillez sélectionner deux variables à explorer dans le menu de gauche')
        st.write('')

        # On va récupérer les colonnes dans une liste
        cols_to_plot = train_data.columns.tolist()
        # On supprime les colonnes à ne pas représenter
        cols_to_plot.remove('SK_ID_CURR')
        cols_to_plot.remove('TARGET')

        st.sidebar.write('')
        selected_var1 = st.sidebar.selectbox('Première variable à explorer', cols_to_plot, index=5)

        # Suppression de la première feature sélectionnée
        cols_to_plot_b = cols_to_plot.copy()
        cols_to_plot_b.remove(selected_var1)

        st.sidebar.write('')
        selected_var2 = st.sidebar.selectbox('Deuxième variable à explorer', cols_to_plot_b, index=10)

        cgrph, ccorr = st.columns(2)
        with cgrph:
            st.write('Pour représenter la relation entre les variables choisies, veuillez cliquer ci-dessous :')
            plot_btn = st.button('Graphique bi-varié')
        with ccorr:
            st.write('Pour la corrélation entre les variables et la cible, veuillez cliquer ci-dessous :')
            corr_btn = st.button('Corrélation')

        # Représentation
        if plot_btn:
            graph_bi = plot_vars(train_data, selected_var1, selected_var2)
            st.plotly_chart(graph_bi, use_container_width=True)

        # Heatmap des corrélations
        if corr_btn:
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
