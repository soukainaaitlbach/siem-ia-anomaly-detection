import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Charger les résultats des 3 modèles
df_if = pd.read_csv('/root/resultats_isolation_forest_no_leakage.csv')
df_ae = pd.read_csv('/root/resultats_autoencoder.csv')
df_lstm = pd.read_csv('/root/resultats_lstm_no_leakage.csv')
# Créer les labels réels
# Level >= 7 = Anomalie, Level < 7 = Normal
df_if['real'] = df_if['level'].apply(lambda x: 'Anomalie' if x >= 7 else 'Normal')
df_ae['real'] = df_ae['level'].apply(lambda x: 'Anomalie' if x >= 7 else 'Normal')
df_lstm['real'] = df_lstm['level'].apply(lambda x: 'Anomalie' if x >= 7 else 'Normal')

# Evaluation Isolation Forest
print("=" * 40)
print("ISOLATION FOREST")
print("=" * 40)
print(classification_report(df_if['real'], df_if['anomaly']))

# Evaluation Autoencoder
print("=" * 40)
print("AUTOENCODER")
print("=" * 40)
print(classification_report(df_ae['real'], df_ae['anomaly']))

# Evaluation LSTM
print("=" * 40)
print("LSTM")
print("=" * 40)
print(classification_report(df_lstm['real'], df_lstm['anomaly']))

# Comparaison finale
print("=" * 40)
print("COMPARAISON FINALE")
print("=" * 40)
models = ['Isolation Forest', 'Autoencoder', 'LSTM']
dfs = [df_if, df_ae, df_lstm]

for model, df in zip(models, dfs):
    acc = accuracy_score(df['real'], df['anomaly'])
    f1 = f1_score(df['real'], df['anomaly'], pos_label='Anomalie', average='weighted')
    print(f"{model}: Accuracy={acc:.2f}, F1-Score={f1:.2f}")

print("\nFichier sauvegardé!")
from sklearn.metrics import confusion_matrix

print("\n" + "=" * 40)
print("TAUX DE FAUX POSITIFS (FPR) PAR MODELE")
print("=" * 40)
for model, df in zip(models, dfs):
    cm = confusion_matrix(df['real'], df['anomaly'], labels=['Normal', 'Anomalie'])
    tn, fp = cm[0][0], cm[0][1]
    fpr = fp / (fp + tn) * 100
    print(f"{model}: {fp} faux positifs sur {fp+tn} logs normaux → FPR = {fpr:.2f}%")
