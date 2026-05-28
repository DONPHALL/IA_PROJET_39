
import streamlit as st
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout="wide")

st.title("Outil de Prédiction d'Abandon de Traitement ARV")
st.write("Cette application aide les agents de terrain à comprendre les facteurs d'abandon et à identifier les patients à risque.")

# --- Chargement du modèle et des données --- #
# Chargement du modèle de Cox entraîné
@st.cache_resource
def load_model():
    return joblib.load('cox_model.joblib')

cph = load_model()

# Chargement du jeu de données d'entraînement complet pour les profils types
@st.cache_data
def load_training_data():
    df = pd.read_csv('df_cox_training.csv')
    # Ensure boolean columns are correctly typed after loading from CSV
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(bool)
    return df

df_cox_training = load_training_data()

# Chargement de l'échantillon de caractéristiques patient pour les entrées par défaut
@st.cache_data
def load_sample_patients():
    df = pd.read_csv('sample_patient_features.csv')
    # Ensure boolean columns are correctly typed after loading from CSV
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(bool)
    return df

df_sample_patients = load_sample_patients()

# Définition des caractéristiques (doit correspondre à celles utilisées pour l'entraînement)
features = ['age', 'initial_cd4', 'initial_viral_load',
            'total_visits', 'avg_visit_interval_days', 'total_sms_sent',
            'total_sms_responded', 'sms_response_rate', 'viral_load_change',
            'gender_M']

# --- Section 1: Résumé du Modèle --- #
st.header("1. Résumé du Modèle de Cox")
st.markdown("Le modèle de Cox évalue l'impact de différentes caractéristiques sur le risque d'abandon de traitement. Un Hazard Ratio (exp(coef)) supérieur à 1 indique une augmentation du risque, tandis qu'un Hazard Ratio inférieur à 1 indique une diminution du risque.")

st.subheader("Coefficients du Modèle")
# Afficher le résumé du modèle de manière lisible
st.write(cph.summary)

st.subheader("Graphique des Coefficients (Log-Hazard Ratios)")
fig1, ax1 = plt.subplots(figsize=(10, 6))
cph.plot(ax=ax1)
ax1.set_title('Coefficients du Modèle de Cox (Log-Hazard Ratios)')
ax1.set_xlabel('Log-Hazard Ratio')
ax1.set_ylabel('Caractéristique')
ax1.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig1)

st.markdown("**Interprétation :** Les barres qui ne traversent pas la ligne verticale à zéro indiquent des facteurs statistiquement significatifs. Un coefficient négatif réduit le risque d'abandon, un positif l'augmente.")

# --- Section 2: Prédiction Personnalisée pour les Patients --- #
st.header("2. Prédiction de Survie pour un Patient Spécifique")
st.markdown("Entrez les caractéristiques d'un patient pour voir sa fonction de survie prédite et évaluer son risque d'abandon.")

# Utilisons les médianes de notre dataset d'entraînement pour les valeurs par défaut des sliders
default_values = df_cox_training[features].median()

with st.expander("Entrer les caractéristiques du patient"): 
    col1, col2, col3 = st.columns(3)

    patient_age = col1.slider('Age', min_value=int(df_cox_training['age'].min()), max_value=int(df_cox_training['age'].max()), value=int(default_values['age']))
    # Handle gender_M as boolean for selectbox
    gender_selection = col1.selectbox('Sexe', options=['Femme', 'Homme'], index=1 if default_values['gender_M'] else 0)
    patient_gender_M = True if gender_selection == 'Homme' else False
    
    patient_initial_cd4 = col1.slider('CD4 Initial', min_value=int(df_cox_training['initial_cd4'].min()), max_value=int(df_cox_training['initial_cd4'].max()), value=int(default_values['initial_cd4']))
    patient_initial_viral_load = col2.slider('Charge Virale Initiale', min_value=int(df_cox_training['initial_viral_load'].min()), max_value=int(df_cox_training['initial_viral_load'].max()), value=int(default_values['initial_viral_load']))
    patient_total_visits = col2.slider('Total Visites', min_value=int(df_cox_training['total_visits'].min()), max_value=int(df_cox_training['total_visits'].max()), value=int(default_values['total_visits']))
    patient_avg_visit_interval_days = col2.slider('Intervalle moyen visites (jours)', min_value=float(df_cox_training['avg_visit_interval_days'].min()), max_value=float(df_cox_training['avg_visit_interval_days'].max()), value=float(default_values['avg_visit_interval_days']))
    patient_total_sms_sent = col3.slider('Total SMS Envoyés', min_value=int(df_cox_training['total_sms_sent'].min()), max_value=int(df_cox_training['total_sms_sent'].max()), value=int(default_values['total_sms_sent']))
    patient_total_sms_responded = col3.slider('Total SMS Répondus', min_value=int(df_cox_training['total_sms_responded'].min()), max_value=int(df_cox_training['total_sms_responded'].max()), value=int(default_values['total_sms_responded']))
    patient_sms_response_rate = col3.slider('Taux Réponse SMS', min_value=float(df_cox_training['sms_response_rate'].min()), max_value=float(df_cox_training['sms_response_rate'].max()), value=float(default_values['sms_response_rate']))
    patient_viral_load_change = col3.slider('Changement Charge Virale', min_value=int(df_cox_training['viral_load_change'].min()), max_value=int(df_cox_training['viral_load_change'].max()), value=int(default_values['viral_load_change']))

    # Créer un DataFrame pour le patient avec les valeurs saisies
    patient_data = pd.DataFrame([[patient_age, patient_initial_cd4, patient_initial_viral_load,
                                  patient_total_visits, patient_avg_visit_interval_days,
                                  patient_total_sms_sent, patient_total_sms_responded,
                                  patient_sms_response_rate, patient_viral_load_change,
                                  patient_gender_M]],
                                columns=features)

# Prédire la fonction de survie pour le patient
sf_patient = cph.predict_survival_function(patient_data)

fig2, ax2 = plt.subplots(figsize=(12, 7))
sf_patient.plot(ax=ax2)
ax2.set_title('Fonction de Survie Prédite pour le Patient')
ax2.set_xlabel('Temps (jours)')
ax2.set_ylabel('Probabilité de Survie')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(['Patient Actuel'])
st.pyplot(fig2)

st.markdown("**Interprétation :** Cette courbe montre la probabilité estimée que le patient continue son traitement au fil du temps. Une courbe qui descend rapidement indique un risque d'abandon plus élevé.")


# --- Section 3: Comparaison de Profils (pour référence) --- #
st.header("3. Comparaison de Fonctions de Survie (Profils Types)")
st.markdown("Comparez le patient actuel avec des profils types (Moyen, Faible Risque, Haut Risque) pour mieux contextualiser le risque.")

# Recréer les profils types en utilisant df_cox_training pour une cohérence avec le modèle entraîné
median_patient_app = df_cox_training[features].median()

low_risk_patient_app = median_patient_app.copy()
low_risk_patient_app['total_visits'] = df_cox_training['total_visits'].quantile(0.75)
low_risk_patient_app['total_sms_responded'] = df_cox_training['total_sms_responded'].quantile(0.75)

high_risk_patient_app = median_patient_app.copy()
high_risk_patient_app['total_visits'] = df_cox_training['total_visits'].quantile(0.25)
high_risk_patient_app['total_sms_responded'] = df_cox_training['total_sms_responded'].quantile(0.25)

# Ajouter le patient actuel au comparatif
patients_to_compare = pd.DataFrame([
    patient_data.iloc[0], # Le patient dont les caractéristiques sont saisies
    median_patient_app,
    low_risk_patient_app,
    high_risk_patient_app
], index=['Patient Actuel', 'Patient Moyen', 'Patient à Faible Risque', 'Patient à Haut Risque'])

# Assurer que gender_M est booléen pour toutes les lignes
patients_to_compare['gender_M'] = patients_to_compare['gender_M'].astype(bool)

sf_comparison = cph.predict_survival_function(patients_to_compare)

fig3, ax3 = plt.subplots(figsize=(12, 7))
sf_comparison.plot(ax=ax3)
ax3.set_title('Fonctions de Survie : Comparaison des Profils')
ax3.set_xlabel('Temps (jours)')
ax3.set_ylabel('Probabilité de Survie')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(title='Profil du Patient')
st.pyplot(fig3)

st.markdown("**Conseil pour l'agent de terrain :** Comparez la courbe du 'Patient Actuel' avec les profils types. Si elle est proche de la courbe 'Patient à Haut Risque', une intervention proactive est fortement recommandée.")
