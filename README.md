# Streamlit web application for Home Credit Default Risk

## Objectif du projet

Le but de ce projet est de réussir à prédire si un client pourra rembourser son crédit, ou s'il y a un risque qu'il soit en défaut de paiement, ceci en vue d'étayer la décision d'accorder, ou non, un prêt à un client potentiel.

## Contenu du dossier

### Requirements

Le fichier requirements.txt contient les packages utilisés et leur version

### Files

Les fichiers train_poc.csv, test_poc.csv, P7_train_norm.csv et P7_test_norm.csv contiennent les données:
- Les fichiers train_poc.csv et test_poc.csv contiennent un échantillon des jeux de données initiaux.
- Les fichiers P7_train_norm.csv et P7_test_norm.csv contiennet ces données précédentes déjà normalisées, prêtes à être utilisées par le modèle.

Le fichier P7_01_analyse.ipynb contient le code du notebook qui a permis l'exploration des données, le traitement des données, ainsi que la génération/export du modèle.
- Note: le dossier importé contenait uniquement les fichiers application_train.csv et application_test.csv, zippés.

Le fichier best_model.pkl contient le modèle entraîné.

### API & dashboard

Le fichier API_predict.py contient le code de l'API, qui est hebergée sur PythonAnywhere.

Le fichier App_viz.py contient le code du dashboard Streamlit, qui est partagé via share.streamlit.

Ce dashboard est accessible via: https://share.streamlit.io/64fc/oc_p7/main/App_viz.py
