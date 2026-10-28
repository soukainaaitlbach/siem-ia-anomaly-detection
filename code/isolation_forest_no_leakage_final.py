import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Chargement des données
df = pd.read_csv('/root/alerts_clean.csv')

print("Données chargées :", df.shape)

# Création des labels de référence
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

# Features utilisées par le modèle
# IMPORTANT : level n'est pas utilisé comme feature
X = df[
    ['agent_encoded', 'location_encoded']
].values

print("Features utilisées :", X.shape)

# Création du modèle
model = IsolationForest(
    contamination=0.1,
    random_state=42
)

# Entraînement
model.fit(X)

print("Modèle entraîné !")

# Prédiction
predictions = model.predict(X)

df['anomaly'] = [
    'Anomalie' if prediction == -1
    else 'Normal'
    for prediction in predictions
]

# Résultats
print("\nRésultats :")
print(df['anomaly'].value_counts())

# Sauvegarde
df.to_csv(
    '/root/resultats_isolation_forest_no_leakage.csv',
    index=False
)

print(
    "\nFichier sauvegardé :",
    '/root/resultats_isolation_forest_no_leakage.csv'
)
