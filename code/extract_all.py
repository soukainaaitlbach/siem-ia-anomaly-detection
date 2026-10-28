import json
import os

all_logs = []
base_paths = ["/var/ossec/logs/alerts", "/var/ossec/logs/archives"]

def classify_source(log):
    groups = log.get("rule", {}).get("groups", [])
    location = log.get("location", "")
    if "suricata" in groups or "ids" in groups:
        return "reseau"
    elif "web" in groups or "accesslog" in groups or "apache" in location:
        return "applicatif"
    else:
        return "systeme"

for base_path in base_paths:
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            log = json.loads(line)
                            log["source_type"] = classify_source(log)
                            all_logs.append(log)
                        except:
                            pass

with open("/root/alerts_extracted.json", "w") as f:
    json.dump(all_logs, f, indent=4)

print(f"{len(all_logs)} logs extraits au total!")
systeme = sum(1 for l in all_logs if l["source_type"] == "systeme")
reseau = sum(1 for l in all_logs if l["source_type"] == "reseau")
appli = sum(1 for l in all_logs if l["source_type"] == "applicatif")
print(f"Système: {systeme} | Réseau: {reseau} | Applicatif: {appli}")
