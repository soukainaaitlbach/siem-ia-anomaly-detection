import pandas as pd

print("=" * 70)
print("COMPARAISON SIEM SEUL vs SIEM + AUTOENCODER")
print("=" * 70)

# ============================================================
# 1. Charger les données
# ============================================================

df = pd.read_csv('/root/resultats_autoencoder.csv')

# ============================================================
# 2. SIEM SEUL
#    Une alerte SIEM = level >= 7
# ============================================================

df['siem'] = df['level'].apply(
    lambda x: 'Anomalie' if x >= 7 else 'Normal'
)

# Nombre d'alertes générées par Wazuh
siem_alertes = (df['siem'] == 'Anomalie').sum()

print("\nSIEM SEUL")
print("-" * 70)
print(f"Nombre d'alertes : {siem_alertes}")

# ============================================================
# 3. SIEM + AUTOENCODER
#
# IMPORTANT :
# On applique l'IA UNIQUEMENT aux alertes SIEM.
# ============================================================

alertes_siem = df[df['siem'] == 'Anomalie'].copy()

# Alertes SIEM conservées par l'IA
alertes_conservees = alertes_siem[
    alertes_siem['anomaly'] == 'Anomalie'
]

# Alertes SIEM filtrées par l'IA
alertes_filtrees = alertes_siem[
    alertes_siem['anomaly'] == 'Normal'
]

n_conservees = len(alertes_conservees)
n_filtrees = len(alertes_filtrees)

print("\nSIEM + AUTOENCODER")
print("-" * 70)
print(f"Alertes SIEM analysées       : {siem_alertes}")
print(f"Alertes conservées           : {n_conservees}")
print(f"Alertes filtrées             : {n_filtrees}")

# ============================================================
# 4. Réduction
# ============================================================

reduction = siem_alertes - n_conservees
reduction_pct = reduction / siem_alertes * 100

print("\nREDUCTION DU VOLUME D'ALERTES")
print("-" * 70)
print(f"SIEM seul                    : {siem_alertes}")
print(f"SIEM + Autoencoder           : {n_conservees}")
print(f"Alertes filtrées             : {reduction}")
print(f"Réduction                    : {reduction_pct:.2f}%")

# ============================================================
# 5. Analyse des alertes filtrées
# ============================================================

print("\nALERTES FILTREES PAR L'AUTOENCODER")
print("-" * 70)

print("\nDescriptions :")
print(
    alertes_filtrees['description']
    .value_counts()
    .head(20)
)

print("\nRépartition par agent :")
print(
    alertes_filtrees['agent']
    .value_counts()
)

print("\nRépartition par location :")
print(
    alertes_filtrees['location']
    .value_counts()
)

# ============================================================
# 6. Vérification finale
# ============================================================

print("\nVERIFICATION")
print("-" * 70)

print(
    f"2588 alertes SIEM = "
    f"{n_conservees} conservées + {n_filtrees} filtrées"
)

print(
    f"{n_conservees} + {n_filtrees} = "
    f"{n_conservees + n_filtrees}"
)

print("=" * 70)
