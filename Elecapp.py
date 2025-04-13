# contenu du app.py extrait du canvas (inclus LSTM + Streamlit + visualisation)
# Prototype complet : Prédiction de coupures d’électricité à La Réunion
# Stack : TensorFlow (LSTM), Streamlit, Pandas, Matplotlib, Scikit-learn

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Chargement des données simulées (remplacer par vos sources réelles)
df = pd.read_csv("donnees_reunion.csv", parse_dates=['datetime'], index_col='datetime')

# 2. Feature Engineering (simplifié)
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
df['rolling_temp'] = df['temperature'].rolling(3).mean()
df['rolling_conso'] = df['consommation'].rolling(3).mean()
df = df.dropna()
# 3. Target : y = coupure (binaire)
y = df['coupure']
X = df.drop(columns=['coupure'])

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split temporel sécurisé
n_samples = len(X_scaled)
n_splits = min(5, n_samples - 1)  # Assure au moins 1 test set

if n_splits < 2:
    st.warning("⚠️ Données insuffisantes pour TimeSeriesSplit — fallback en train/test split classique.")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=False)
else:
    ts = TimeSeriesSplit(n_splits=n_splits)
    train_idx, test_idx = list(ts.split(X_scaled))[-1]
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# 5. Mise en forme pour LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 6. Modèle TensorFlow LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

# 6bis. Prédiction sur toute la série temporelle pour l'animation
X_all_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
y_pred_proba = model.predict(X_all_lstm).flatten()

df['proba_coupure'] = y_pred_proba


# 7. Interface Streamlit
st.title("Prédiction de Coupures d'électricité - La Réunion")
st.markdown("Cette application prédit les coupures d'électricité à partir de données météo, consommation et calendrier.")

# Dashboard
fig, ax = plt.subplots()
sns.lineplot(data=df[['consommation', 'temperature']], ax=ax)
ax.set_title("Consommation & Température")
st.pyplot(fig)

# Prédiction interactive
st.subheader("Simulation de Scénario")
temp_input = st.slider("Température", min_value=15.0, max_value=40.0, step=0.5)
conso_input = st.slider("Consommation (MW)", min_value=50.0, max_value=500.0, step=10.0)
hour_input = st.slider("Heure", 0, 23)
is_weekend = st.checkbox("Weekend")

# Approximation de dayofweek à partir de l'heure (facultatif mais nécessaire pour scaler)
# Idéalement, il faudrait demander la date complète pour le vrai dayofweek
dayofweek_input = hour_input % 7  # Juste une estimation

# Approximation pour rolling_temp et rolling_conso (valeurs "instantanées" moyennées)
rolling_temp = temp_input
rolling_conso = conso_input

# Respecte l'ordre exact des colonnes sur lesquelles le scaler a été fit
# Prédiction interactive

st.subheader("Simulation de Scénario")

# Sliders (avec clés uniques pour éviter les erreurs de duplication)
temp_input = st.slider("Température", min_value=15.0, max_value=40.0, step=0.5, key="temp_slider")
conso_input = st.slider("Consommation (MW)", min_value=50.0, max_value=500.0, step=10.0, key="conso_slider")
hour_input = st.slider("Heure", 0, 23, key="hour_slider")
is_weekend = st.checkbox("Weekend", key="weekend_checkbox")

# Approximation des features manquants
dayofweek_input = hour_input % 7
rolling_temp = temp_input
rolling_conso = conso_input

# Construction de l'entrée avec les 7 features dans le bon ordre
input_data = scaler.transform([[
    temp_input,         # temperature
    conso_input,        # consommation
    hour_input,         # hour
    dayofweek_input,    # dayofweek
    int(is_weekend),    # is_weekend
    rolling_temp,       # rolling_temp
    rolling_conso       # rolling_conso
]])

# Préparation pour LSTM
input_lstm = input_data.reshape((1, 1, input_data.shape[1]))
proba = model.predict(input_lstm)[0][0]

# Affichage
st.metric("Probabilité de coupure", f"{proba*100:.2f}%")


# Préparation pour LSTM
input_lstm = input_data.reshape((1, 1, input_data.shape[1]))
proba = model.predict(input_lstm)[0][0]


st.metric("Probabilité de coupure", f"{proba*100:.2f}%")

# Gadgets à intégrer (non codés ici)
st.markdown("""
- **Synthèse IA** : rapport automatique des risques (exemple : *"Attention, risque de coupure dans l'ouest à 18h"*).
- **Alertes connectées** : via SMS, LED, assistant vocal.
- **Cartographie en temps réel** (prochaine étape).
- **Explication IA** : affichage des facteurs principaux influents.
""")
st.subheader("📊 Évolution de la probabilité de coupure")

# Slider temporel interactif
time_index = st.slider("Choisir une position dans le temps", 0, len(df)-1, len(df)-1, 1)

fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.lineplot(x=df.index[:time_index+1], y=df['proba_coupure'][:time_index+1], ax=ax2, color='crimson', label='Probabilité de coupure')
ax2.set_title("Probabilité de coupure dans le temps")
ax2.set_ylabel("Proba (0 → 1)")
ax2.set_xlabel("Temps")
ax2.axhline(0.5, ls='--', color='gray', label='Seuil 50%')
ax2.legend()
st.pyplot(fig2)

import time

if st.button("▶ Lancer l'animation"):
    for i in range(10, len(df), 10):
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=df.index[:i], y=df['proba_coupure'][:i], ax=ax3, color='crimson')
        ax3.set_title("Animation de la probabilité de coupure")
        ax3.set_ylabel("Probabilité")
        ax3.set_xlabel("Temps")
        st.pyplot(fig3)
        time.sleep(0.1)
