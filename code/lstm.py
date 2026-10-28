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

# Encodage one-hot de source_type (systeme/reseau/applicatif)
df = pd.get_dummies(df, columns=['source_type'], prefix='src')

# IMPORTANT :
# level n'est pas utilisé comme feature (évite le data leakage)
feature_cols = ['agent_encoded', 'location_encoded', 'nb_groups']
feature_cols += [c for c in df.columns if c.startswith('src_')]

X = df[feature_cols].values.astype(float)
n_features = X.shape[1]

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
        shape=(timesteps, n_features)
    ),
    layers.LSTM(
        64,
        return_sequences=True
    ),
    layers.Dropout(0.2),
    layers.LSTM(
        32
    ),
    layers.Dense(n_features)
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
