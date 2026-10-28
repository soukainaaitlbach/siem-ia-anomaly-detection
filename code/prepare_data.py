import json
import pandas as pd
import numpy as np

with open('/root/alerts_extracted.json') as f:
    logs = json.load(f)

data = []
for log in logs:
    row = {
        'level': log.get('rule', {}).get('level', 0),
        'description': log.get('rule', {}).get('description', ''),
        'agent': log.get('agent', {}).get('name', ''),
        'timestamp': log.get('timestamp', ''),
        'location': log.get('location', ''),
        'source_type': log.get('source_type', 'systeme'),
        'nb_groups': len(log.get('rule', {}).get('groups', []))
    }
    data.append(row)

df = pd.DataFrame(data)
print(df.head(10))
print("\nShape:", df.shape)
print("\nNiveaux uniques:", df['level'].unique())
print("\nRepartition source_type:")
print(df['source_type'].value_counts())

df.to_csv('/root/alerts_clean.csv', index=False)
print("\nFichier CSV sauvegarde!")
