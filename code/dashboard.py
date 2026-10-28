import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# Charger les données réelles (fichiers actuels)
df_if = pd.read_csv('/root/resultats_isolation_forest_no_leakage.csv')
df_ae = pd.read_csv('/root/resultats_autoencoder.csv')
df_lstm = pd.read_csv('/root/resultats_lstm_no_leakage.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dashboard PFE - SIEM + IA\nDétection d\'Anomalies', fontsize=16, fontweight='bold')

# Graphique 1 - Comparaison des 3 modèles (calculé dynamiquement)
models = ['Isolation Forest', 'Autoencoder', 'LSTM']
normaux = [
    (df_if['anomaly'] == 'Normal').sum(),
    (df_ae['anomaly'] == 'Normal').sum(),
    (df_lstm['anomaly'] == 'Normal').sum()
]
anomalies = [
    (df_if['anomaly'] == 'Anomalie').sum(),
    (df_ae['anomaly'] == 'Anomalie').sum(),
    (df_lstm['anomaly'] == 'Anomalie').sum()
]
x = np.arange(len(models))
width = 0.35
axes[0,0].bar(x - width/2, normaux, width, label='Normal', color='green', alpha=0.7)
axes[0,0].bar(x + width/2, anomalies, width, label='Anomalie', color='red', alpha=0.7)
axes[0,0].set_title('Comparaison des 3 modèles IA')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(models)
axes[0,0].legend()
axes[0,0].set_ylabel('Nombre de logs')

# Graphique 2 - Score de risque (distribution des ia_score depuis le pipeline temps réel)
scores = []
try:
    with open('/var/log/ia_verdict.log') as f:
        for line in f:
            try:
                d = json.loads(line)
                scores.append(d['ia_score'])
            except:
                pass
except FileNotFoundError:
    scores = []

if scores:
    scores_arr = np.array(scores)
    risk_scores = (scores_arr / scores_arr.max() * 100) if scores_arr.max() > 0 else scores_arr
    axes[0,1].hist(risk_scores, bins=20, color='darkorange', alpha=0.7)
    axes[0,1].set_title(f'Distribution des scores de risque (0-100)\n{len(scores)} événements temps réel')
    axes[0,1].set_xlabel('Score de risque')
    axes[0,1].set_ylabel('Nombre d\'événements')
else:
    axes[0,1].text(0.5, 0.5, 'Pas encore de données\ndu pipeline temps réel', ha='center', va='center')
    axes[0,1].set_title('Distribution des scores de risque')

# Graphique 3 - SIEM seul vs SIEM+IA (calculé dynamiquement)
siem_seul_count = (df_ae['level'] >= 7).sum()
ia_count = (df_ae['anomaly'] == 'Anomalie').sum()
categories = ['SIEM seul', 'SIEM + IA']
anomalies_comp = [siem_seul_count, ia_count]
colors_comp = ['red' if ia_count > siem_seul_count else 'orange', 'green' if ia_count <= siem_seul_count else 'red']
bars2 = axes[1,0].bar(categories, anomalies_comp, color=colors_comp, alpha=0.7)
axes[1,0].set_title('SIEM seul vs SIEM + IA (volume d\'alertes)')
axes[1,0].set_ylabel('Nombre d\'anomalies')
for bar, val in zip(bars2, anomalies_comp):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold')

# Graphique 4 - Distribution des niveaux
df_ae['level'].value_counts().sort_index().plot(kind='bar', ax=axes[1,1], color='purple', alpha=0.7)
axes[1,1].set_title('Distribution des niveaux d\'alerte')
axes[1,1].set_xlabel('Niveau')
axes[1,1].set_ylabel('Nombre')
axes[1,1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('/root/dashboard_pfe.png', dpi=150, bbox_inches='tight')
print("Dashboard sauvegardé: /root/dashboard_pfe.png")
