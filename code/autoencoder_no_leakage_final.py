import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Chargement des données
df = pd.read_csv('/root/alerts_clean.csv')

# Labels de référence
# level >= 7 : Anomalie
# level < 7  : Normal
df['real'] = df['level'].apply(
    lambda x: 'Anomalie' if x >= 7 else 'Normal'
)

# Encodage des variables catégorielles
le_agent = LabelEncoder()
le_location = LabelEncoder()

df['agent_encoded'] = le_agent.fit_transform(
    df['agent'].astype(str)
)

df['location_encoded'] = le_location.fit_transform(
    df['location'].astype(str)
)

# IMPORTANT :
# level n'est pas utilisé comme feature
X = df[
    ['agent_encoded', 'location_encoded']
].values

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sélection des données normales uniquement
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

# Entraînement uniquement sur les données normales
autoencoder.fit(
    X_normal,
    X_normal,
    epochs=50,
    batch_size=32,
    verbose=0
)

print("Modèle Autoencoder entraîné !")

# Calcul du seuil sur les données normales
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
    95
)

print(
    "Seuil de détection :",
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

# Résultats
print("\nRésultats :")
print(
    df['anomaly'].value_counts()
)

# Sauvegarde
df.to_csv(
    '/root/resultats_autoencoder_no_leakage.csv',
    index=False
)

print(
    "\nFichier sauvegardé :",
    '/root/resultats_autoencoder_no_leakage.csv'
)
