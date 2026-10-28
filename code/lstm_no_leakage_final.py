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

# Création des séquences
timesteps = 10
X_seq = []

for i in range(len(X_scaled) - timesteps):
    X_seq.append(
        X_scaled[i:i + timesteps]
    )

X_seq = np.array(X_seq)

# Construction du modèle
model = keras.Sequential([
    layers.Input(
        shape=(timesteps, 2)
    ),

    layers.LSTM(
        64,
        return_sequences=True
    ),

    layers.Dropout(0.2),

    layers.LSTM(
        32
    ),

    layers.Dense(2)
])

# Compilation
model.compile(
    optimizer='adam',
    loss='mse'
)

# Entraînement
model.fit(
    X_seq,
    X_scaled[timesteps:],
    epochs=50,
    batch_size=32,
    verbose=0
)

print("Modèle LSTM entraîné !")

# Prédictions
predictions = model.predict(
    X_seq,
    verbose=0
)

# Erreur de prédiction
mse = np.mean(
    np.power(
        X_scaled[timesteps:] - predictions,
        2
    ),
    axis=1
)

# Seuil
threshold = np.percentile(
    mse,
    90
)

print(
    "Seuil de détection :",
    threshold
)

# Détection
results = [
    'Anomalie' if error > threshold
    else 'Normal'
    for error in mse
]

# Résultats
print("\nRésultats :")
print(
    pd.Series(results).value_counts()
)

# Préparation du fichier de sortie
df_result = df.iloc[timesteps:].copy()

df_result['anomaly'] = results

# Sauvegarde
df_result.to_csv(
    '/root/resultats_lstm_no_leakage.csv',
    index=False
)

print(
    "\nFichier sauvegardé :",
    '/root/resultats_lstm_no_leakage.csv'
)
