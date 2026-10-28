import json

input_file = "/var/ossec/logs/alerts/alerts.json"

output_file = "/root/alerts_extracted.json"

logs = []

with open(input_file, "r", encoding="utf-8") as f:

    for line in f:

        try:

            logs.append(json.loads(line))

        except:

            pass

with open(output_file, "w", encoding="utf-8") as f:

    json.dump(logs, f, indent=4)

print(f"{len(logs)} logs extraits.")
