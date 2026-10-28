import pandas as pd
import numpy as np
from tensorflow import keras
import pickle

df = pd.read_csv('/root/alerts_clean.csv')
with open('/root/autoencoder_extras.pkl', 'rb') as f:
    extras = pickle.load(f)
scaler = extras['scaler']
le_agent = extras['le_agent']
le_location = extras['le_location']
feature_cols = extras['feature_cols']

autoencoder = keras.models.load_model('/root/autoencoder_model.keras')

df['agent_encoded'] = le_agent.transform(df['agent'].astype(str))
df['location_encoded'] = le_location.transform(df['location'].astype(str))
df = pd.get_dummies(df, columns=['source_type'], prefix='src')
df['real'] = df['level'].apply(lambda x: 'Anomalie' if x >= 7 else 'Normal')

X = df[feature_cols].values.astype(float)
X_scaled = scaler.transform(X)
reconstruction = autoencoder.predict(X_scaled, verbose=0)
mse = np.mean(np.power(X_scaled - reconstruction, 2), axis=1)

mse_normal = mse[df['real'] == 'Normal']

n_siem_seul = (df['real'] == 'Anomalie').sum()

print(f"Référence SIEM seul : {n_siem_seul} anomalies\n")

for p in [95, 98, 99, 99.5]:
    threshold = np.percentile(mse_normal, p)
    n_detectees = (mse > threshold).sum()
    variation = (n_detectees - n_siem_seul) / n_siem_seul * 100
    print(f"Percentile {p}e : seuil={threshold:.6f} | Anomalies détectées={n_detectees} | Variation vs SIEM seul={variation:+.1f}%")
