import json
import time
import pickle
import numpy as np
from tensorflow import keras

# Chargement du modèle et des paramètres sauvegardés
autoencoder = keras.models.load_model('/root/autoencoder_model.keras')
with open('/root/autoencoder_extras.pkl', 'rb') as f:
    extras = pickle.load(f)

scaler = extras['scaler']
le_agent = extras['le_agent']
le_location = extras['le_location']
threshold = extras['threshold']
feature_cols = extras['feature_cols']

def classify_source(log):
    groups = log.get("rule", {}).get("groups", [])
    location = log.get("location", "")
    if "suricata" in groups or "ids" in groups:
        return "reseau"
    elif "web" in groups or "accesslog" in groups or "apache" in location:
        return "applicatif"
    else:
        return "systeme"

def safe_encode(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return -1

print("Pipeline temps réel démarré, surveillance de alerts.json...")

filepath = "/var/ossec/logs/alerts/alerts.json"
with open(filepath, "r") as f:
    f.seek(0, 2)  # se placer à la fin du fichier
    while True:
        line = f.readline()
        if not line:
            time.sleep(1)
            continue
        try:
            log = json.loads(line)
        except:
            continue

        source_type = classify_source(log)
        agent = log.get("agent", {}).get("name", "")
        location = log.get("location", "")
        nb_groups = len(log.get("rule", {}).get("groups", []))

        row = {
               'level': log.get('rule', {}).get('level', 0),
               'agent_encoded': safe_encode(le_agent, agent),
               'location_encoded': safe_encode(le_location, location),
               'nb_groups': nb_groups,
               'src_applicatif': 1 if source_type == "applicatif" else 0,
               'src_reseau': 1 if source_type == "reseau" else 0,
               'src_systeme': 1 if source_type == "systeme" else 0,
               }
        X = np.array([[row.get(col, 0) for col in feature_cols]], dtype=float)
        X_scaled = scaler.transform(X)
        reconstruction = autoencoder.predict(X_scaled, verbose=0)
        error = np.mean(np.power(X_scaled - reconstruction, 2))

        verdict = "Anomalie" if error > threshold else "Normal"

        result = {
            "timestamp": log.get("timestamp", ""),
            "agent": agent,
            "source_type": source_type,
            "ia_verdict": verdict,
            "ia_score": float(error)
        }

        with open("/var/log/ia_verdict.log", "a") as out:
            out.write(json.dumps(result) + "\n")

        if verdict == "Anomalie":
            print(f"[ANOMALIE] {agent} | {source_type} | score={error:.6f}")
