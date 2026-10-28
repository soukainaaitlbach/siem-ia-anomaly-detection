import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
# Chargement des données
df = pd.read_csv('/root/alerts_clean.csv')

# Encodage des variables catégorielles
le_agent = LabelEncoder()
le_location = LabelEncoder()

df['agent_encoded'] = le_agent.fit_transform(
    df['agent'].astype(str)
)

df['location_encoded'] = le_location.fit_transform(
    df['location'].astype(str)
)

# Définition des labels réels
# level >= 7 : Anomalie
# level < 7  : Normal
df['real'] = df['level'].apply(
    lambda x: 'Anomalie' if x >= 7 else 'Normal'
)

# Encodage one-hot de source_type (systeme/reseau/applicatif)
df = pd.get_dummies(df, columns=['source_type'], prefix='src')

# Préparation des données (colonnes de base + nouvelles colonnes source_type)
feature_cols = ['level', 'agent_encoded', 'location_encoded', 'nb_groups']
feature_cols += [c for c in df.columns if c.startswith('src_')]

X = df[feature_cols].values.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sélection des données normales
X_normal = X_scaled[
    df['real'] == 'Normal'
]

print("Nombre total de logs :", len(X_scaled))
print("Nombre de logs normaux :", len(X_normal))
print(
    "Nombre de logs anormaux :",
    len(X_scaled) - len(X_normal)
)

# Construction de l'Autoencoder
input_dim = X_scaled.shape[1]

input_layer = keras.Input(
    shape=(input_dim,)
)

encoded = layers.Dense(
    8,
    activation='relu'
)(input_layer)

encoded = layers.Dense(
    4,
    activation='relu'
)(encoded)

decoded = layers.Dense(
    8,
    activation='relu'
)(encoded)

decoded = layers.Dense(
    input_dim,
    activation='linear'
)(decoded)

autoencoder = keras.Model(
    input_layer,
    decoded
)

# Compilation
autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

# Entraînement sur les données normales
autoencoder.fit(
    X_normal,
    X_normal,
    epochs=50,
    batch_size=32,
    verbose=0
)

print("Modele entraine !")

# Calcul du seuil
reconstructions_normal = autoencoder.predict(
    X_normal,
    verbose=0
)

mse_normal = np.mean(
    np.power(
        X_normal - reconstructions_normal,
        2
    ),
    axis=1
)

threshold = np.percentile(
    mse_normal,
    99.5
)

print(
    "Seuil de detection :",
    threshold
)

# Reconstruction de tous les logs
reconstructions = autoencoder.predict(
    X_scaled,
    verbose=0
)

mse = np.mean(
    np.power(
        X_scaled - reconstructions,
        2
    ),
    axis=1
)

# Détection des anomalies
df['anomaly'] = [
    'Anomalie' if error > threshold
    else 'Normal'
    for error in mse
]

# Affichage des résultats
print("Resultats :")
print(
    df['anomaly'].value_counts()
)

# Sauvegarde
df.to_csv(
    '/root/resultats_autoencoder.csv',
    index=False
)

print(
    "Fichier sauvegarde :",
    "/root/resultats_autoencoder.csv"
)
import pickle

autoencoder.save('/root/autoencoder_model.keras')
with open('/root/autoencoder_extras.pkl', 'wb') as f:
    pickle.dump({
        'scaler': scaler,
        'le_agent': le_agent,
        'le_location': le_location,
        'threshold': threshold,
        'feature_cols': feature_cols
    }, f)

print("Modèle et paramètres sauvegardés pour le pipeline temps réel !")
