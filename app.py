
import streamlit as st
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Outil de Prédiction d'Abandon de Traitement ARV")
st.write("Cette application aide les agents de terrain à comprendre les facteurs d'abandon et à identifier les patients à risque.")

# --- Chargement et Préparation des Données (simulées pour l'exemple Streamlit) ---

# NOTE: Dans une application réelle, vous chargereriez ici les données réelles et le modèle entraîné.
# Pour cet exemple, nous allons recréer une simulation simplifiée des données et du modèle.

# Paramètres de simulation simplifiés pour Streamlit
n_patients_sim = 200 # Moins de patients pour un chargement rapide
start_date_sim = pd.to_datetime('2020-01-01')
end_date_sim = pd.to_datetime('2023-12-31')

# Données patients simplifiées
patients_sim = {
    'patient_id': range(1, n_patients_sim + 1),
    'age': np.random.randint(18, 70, n_patients_sim),
    'gender': np.random.choice(['M', 'F'], n_patients_sim, p=[0.5, 0.5]),
    'enrollment_date': [start_date_sim + pd.Timedelta(days=np.random.randint(0, (end_date_sim - start_date_sim).days)) for _ in range(n_patients_sim)],
    'initial_cd4': np.random.randint(100, 800, n_patients_sim),
    'initial_viral_load': np.random.randint(50, 100000, n_patients_sim)
}
df_patients_sim = pd.DataFrame(patients_sim)

# Simulation d'événements et de visites (simplifiée pour l'exemple Streamlit)
# Dans une vraie app, ces données seraient chargées depuis une DB ou un fichier.
# Nous allons juste simuler des T et E pour avoir un df_cox minimal.
df_patients_sim['event_date'] = df_patients_sim['enrollment_date'] + pd.to_timedelta(np.random.randint(180, 1000, n_patients_sim), unit='D')
df_patients_sim['event_status'] = np.random.choice([0, 1], n_patients_sim, p=[0.8, 0.2])
df_patients_sim['last_known_visit_date'] = df_patients_sim['event_date'] - pd.to_timedelta(np.random.randint(30, 90, n_patients_sim), unit='D')

# Simuler quelques features pour df_cox
df_patients_sim['total_visits'] = np.random.randint(1, 20, n_patients_sim)
df_patients_sim['avg_visit_interval_days'] = np.random.uniform(10, 100, n_patients_sim)
df_patients_sim['total_sms_sent'] = np.random.randint(0, 30, n_patients_sim)
df_patients_sim['total_sms_responded'] = df_patients_sim['total_sms_sent'] * np.random.uniform(0.5, 1.0, n_patients_sim)
df_patients_sim['sms_response_rate'] = df_patients_sim['total_sms_responded'] / (df_patients_sim['total_sms_sent'] + 1e-6)
df_patients_sim['viral_load_change'] = np.random.randint(-50000, 50000, n_patients_sim)

df_model_sim = df_patients_sim.copy()
df_model_sim['T'] = (df_model_sim['event_date'] - df_model_sim['enrollment_date']).dt.days
df_model_sim['E'] = df_model_sim['event_status']

features_sim = ['age', 'initial_cd4', 'initial_viral_load',
                'total_visits', 'avg_visit_interval_days', 'total_sms_sent',
                'total_sms_responded', 'sms_response_rate', 'viral_load_change']

df_model_sim = pd.get_dummies(df_model_sim, columns=['gender'], drop_first=True)
if 'gender' in features_sim: features_sim.remove('gender')
features_sim.append('gender_M')

df_cox_sim = df_model_sim[features_sim + ['T', 'E']]
df_cox_sim = df_cox_sim.dropna()

cph_sim = CoxPHFitter(penalizer=0.1)
cph_sim.fit(df_cox_sim, duration_col='T', event_col='E', show_progress=False)


# --- Section 1: Résumé du Modèle ---
st.header("1. Résumé du Modèle de Cox")
st.markdown("Le modèle de Cox évalue l'impact de différentes caractéristiques sur le risque d'abandon de traitement. Un Hazard Ratio (exp(coef)) supérieur à 1 indique une augmentation du risque, tandis qu'un Hazard Ratio inférieur à 1 indique une diminution du risque.")

st.subheader("Coefficients du Modèle")
# Afficher le résumé du modèle de manière lisible
st.write(cph_sim.summary)

st.subheader("Graphique des Coefficients (Log-Hazard Ratios)")
fig1, ax1 = plt.subplots(figsize=(10, 6))
cph_sim.plot(ax=ax1)
ax1.set_title('Coefficients du Modèle de Cox (Log-Hazard Ratios)')
ax1.set_xlabel('Log-Hazard Ratio')
ax1.set_ylabel('Caractéristique')
ax1.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig1)

st.markdown("**Interprétation :** Les barres qui ne traversent pas la ligne verticale à zéro indiquent des facteurs statistiquement significatifs. Un coefficient négatif réduit le risque d'abandon, un positif l'augmente.")

# --- Section 2: Prédiction Personnalisée pour les Patients ---
st.header("2. Prédiction de Survie pour un Patient Spécifique")
st.markdown("Entrez les caractéristiques d'un patient pour voir sa fonction de survie prédite et évaluer son risque d'abandon.")

# Créer des sliders ou entrées pour les caractéristiques
# Utilisons les moyennes/médianes comme valeurs par défaut pour les entrées

default_values = df_cox_sim[features_sim].median()

with st.expander("Entrer les caractéristiques du patient"):
    col1, col2, col3 = st.columns(3)

    patient_age = col1.slider('Age', min_value=18, max_value=80, value=int(default_values['age']))
    patient_gender_M = col1.selectbox('Sexe', options=[False, True], format_func=lambda x: 'Homme' if x else 'Femme', index=0 if default_values['gender_M'] == 0 else 1)
    patient_initial_cd4 = col1.slider('CD4 Initial', min_value=100, max_value=800, value=int(default_values['initial_cd4']))
    patient_initial_viral_load = col2.slider('Charge Virale Initiale', min_value=50, max_value=100000, value=int(default_values['initial_viral_load']))
    patient_total_visits = col2.slider('Total Visites', min_value=0, max_value=50, value=int(default_values['total_visits']))
    patient_avg_visit_interval_days = col2.slider('Intervalle moyen visites (jours)', min_value=0, max_value=300, value=int(default_values['avg_visit_interval_days']))
    patient_total_sms_sent = col3.slider('Total SMS Envoyés', min_value=0, max_value=50, value=int(default_values['total_sms_sent']))
    patient_total_sms_responded = col3.slider('Total SMS Répondus', min_value=0, max_value=50, value=int(default_values['total_sms_responded']))
    patient_sms_response_rate = col3.slider('Taux Réponse SMS', min_value=0.0, max_value=1.0, value=float(default_values['sms_response_rate']))
    patient_viral_load_change = col3.slider('Changement Charge Virale', min_value=-50000, max_value=50000, value=int(default_values['viral_load_change']))

    # Créer un DataFrame pour le patient avec les valeurs par défaut ou saisies
    patient_data = pd.DataFrame([[patient_age, patient_initial_cd4, patient_initial_viral_load,
                                  patient_total_visits, patient_avg_visit_interval_days,
                                  patient_total_sms_sent, patient_total_sms_responded,
                                  patient_sms_response_rate, patient_viral_load_change,
                                  patient_gender_M]],
                                columns=features_sim)

# Prédire la fonction de survie pour le patient
sf_patient = cph_sim.predict_survival_function(patient_data)

fig2, ax2 = plt.subplots(figsize=(12, 7))
sf_patient.plot(ax=ax2)
ax2.set_title('Fonction de Survie Prédite pour le Patient')
ax2.set_xlabel('Temps (jours)')
ax2.set_ylabel('Probabilité de Survie')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(['Patient Actuel'])
st.pyplot(fig2)

st.markdown("**Interprétation :** Cette courbe montre la probabilité estimée que le patient continue son traitement au fil du temps. Une courbe qui descend rapidement indique un risque d'abandon plus élevé.")


# --- Section 3: Comparaison de Profils (pour référence) ---
st.header("3. Comparaison de Fonctions de Survie (Profils Types)")
st.markdown("Comparez le patient actuel avec des profils types (Moyen, Faible Risque, Haut Risque) pour mieux contextualiser le risque.")

# Recréer les profils types pour la simulation
median_patient_sim = df_cox_sim[features_sim].median()
low_risk_patient_sim = median_patient_sim.copy()
low_risk_patient_sim['total_visits'] = df_cox_sim['total_visits'].quantile(0.75)
low_risk_patient_sim['total_sms_responded'] = df_cox_sim['total_sms_responded'].quantile(0.75)

high_risk_patient_sim = median_patient_sim.copy()
high_risk_patient_sim['total_visits'] = df_cox_sim['total_visits'].quantile(0.25)
high_risk_patient_sim['total_sms_responded'] = df_cox_sim['total_sms_responded'].quantile(0.25)

# Assurer que gender_M est booléen
low_risk_patient_sim['gender_M'] = low_risk_patient_sim['gender_M'].astype(bool)
high_risk_patient_sim['gender_M'] = high_risk_patient_sim['gender_M'].astype(bool)

# Ajouter le patient actuel au comparatif
patients_to_compare = pd.DataFrame([
    patient_data.iloc[0], # Le patient dont les caractéristiques sont saisies
    median_patient_sim,
    low_risk_patient_sim,
    high_risk_patient_sim
], index=['Patient Actuel', 'Patient Moyen', 'Patient à Faible Risque', 'Patient à Haut Risque'])

# Assurer que gender_M est booléen pour toutes les lignes
patients_to_compare['gender_M'] = patients_to_compare['gender_M'].astype(bool)

sf_comparison = cph_sim.predict_survival_function(patients_to_compare)

fig3, ax3 = plt.subplots(figsize=(12, 7))
sf_comparison.plot(ax=ax3)
ax3.set_title('Fonctions de Survie : Comparaison des Profils')
ax3.set_xlabel('Temps (jours)')
ax3.set_ylabel('Probabilité de Survie')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(title='Profil du Patient')
st.pyplot(fig3)

st.markdown("**Conseil pour l'agent de terrain :** Comparez la courbe du 'Patient Actuel' avec les profils types. Si elle est proche de la courbe 'Patient à Haut Risque', une intervention proactive est fortement recommandée.")
