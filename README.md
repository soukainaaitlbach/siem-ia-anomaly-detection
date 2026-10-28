# Intelligent SIEM: AI-Based Anomaly Detection with Wazuh

Academic final-year project for the Master's degree in Artificial Intelligence and Cybersecurity. The project explores combining Wazuh security alerts with machine-learning methods to identify unusual events.

## Objectives

- Collect and inspect security events generated in a Wazuh environment.
- Prepare alert data for machine-learning experiments.
- Compare Isolation Forest, Autoencoder, and LSTM approaches.
- Evaluate detections and visualize model results.
- Explore a streaming pipeline that scores newly generated Wazuh alerts.

## Technologies

- Wazuh SIEM and Wazuh JSON alerts
- Python, pandas, NumPy, scikit-learn
- TensorFlow/Keras
- Matplotlib
- Optional lab components: ELK, Suricata, Windows Server and Debian agents

## Repository layout

```text
siem-ia-anomaly-detection/
├── code/       # Extraction, preprocessing, models, evaluation and dashboard
├── models/     # Saved Autoencoder and preprocessing metadata
├── data/       # Place your locally collected/approved dataset here (not committed)
├── results/    # Generated experiment outputs (not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset and privacy

Raw security logs and experiment-result CSV files are intentionally not included in this public repository. They may contain host names, timestamps, event details, or other environment-specific information. Use only logs you are authorized to process, sanitize them before sharing, and keep local data under `data/`.

The scripts were originally developed in a Linux lab environment and may require path adjustments for your own setup. Wazuh live monitoring also requires access to the local Wazuh alerts file and appropriate operating-system permissions.

## Installation

Use Python in a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Main scripts

- `code/extract.py` and `code/extract_all.py`: extract Wazuh alerts.
- `code/prepare_data.py`: prepare the dataset.
- `code/isolation_forest_no_leakage_final.py`: Isolation Forest experiment.
- `code/autoencoder_no_leakage_final.py`: Autoencoder experiment.
- `code/lstm_no_leakage_final.py`: LSTM experiment.
- `code/evaluation.py` and `code/comparaison_finale.py`: compare model results.
- `code/etude_sensibilite.py`: threshold sensitivity study.
- `code/dashboard.py`: results visualization.
- `code/pipeline_temps_reel.py`: prototype for scoring new Wazuh alerts.

Check the paths and dataset schema in each script before running it. The live pipeline is a lab prototype, not a production-ready security control.

## Reported experimental results

The final report recorded the following Autoencoder metrics on the project's evaluation dataset:

| Metric | Result |
|---|---:|
| Accuracy | 99.23% |
| Precision | 86.97% |
| Recall | 91.81% |
| False-positive rate | 0.50% |

These figures describe the project's own experiment and are not a guarantee of performance on other environments.

## Author

Soukaina Ait-Lbach — Master's project in Artificial Intelligence and Cybersecurity.
