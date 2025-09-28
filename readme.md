## 🚫📧Spam Email Classification

A production-ready Spam Email Classification project with a modular codebase (ingestion → preprocessing → features → training → evaluation), Jupyter notebooks for EDA and clearance steps, logging & reports, and reproducible pipeline orchestration using DVC.

## 📌Project Summary

This repository implements a full ML workflow to detect spam emails.
It includes data ingestion, preprocessing, feature engineering, model training, evaluation, experiment tracking (via DVC), and generated logs/reports for audits and review.

.....................

Primary goals:
- Reproducible pipeline for spam detection
- Clear separation of steps (each stage is a separate file/module)
- Experiment tracking and remote storage using DVC
- Easy-to-run notebooks for EDA / clearance / demos

## 🔑Key Features

- Modular source code: ingestion → preprocessing → features → training → evaluation
- Jupyter notebooks for EDA and clearance checks
- Logging (logs/) and human-readable reports (reports/) produced automatically
- DVC pipeline (dvc.yaml) to reproduce entire workflow
- Model artifact storage (via DVC remote) and versioned experiments
- Automated tests (optional) to validate pipeline components

## 📂Repo structure

Email-Spam-Classification/
│-- src/
│   ├── ingestion/
│   │   └── ingest.py            # Read external data sources / load raw data
│   ├── preprocessing/
│   │   └── preprocess.py        # Cleaning, text normalization, tokenization
│   ├── features/
│   │   └── build_features.py    # Vectorization (TF-IDF), embeddings,
│   ├── training/
│   │   └── train.py             # Model training & save model (joblib/pickles)
│   ├── evaluation/
│   │   └── evaluate.py          # Metrics, confusion matrix, ROC, PR curves
│   
│
│-- notebooks/
│   └── 03_experiments.ipynb     # Experiment walkthroughs
│
│-- data/                        # (gitignored) raw and intermediate datasets
│-- models/                      # (gitignored) saved model artifacts
│-- reports/                     # (gitignored) generated reports, figures
│-- logs/                        # (gitignored) run logs
│-- dvc.yaml                     # DVC pipeline stages
│-- params.yaml                  # Pipeline parameters (learning rate, etc.)
│-- .gitignore
│-- requirements.txt
│-- README.md

## ▶️ How to run the pipeline (reproducible)

# reproduce all stages (ingest → preprocess → features → train → evaluate)
dvc repro
dvc dag
dvc repro train

## 🧪Notebooks & Clearance

- notebooks/03_experiments.ipynb — Quick experiments and visualization of metrics

Use notebooks for human review and to generate initial reports. Keep results reproducible by preferring the DVC pipeline for production runs.

## 📊Logging & Reports

- All runtime logs are written to logs/
- Evaluation outputs and charts saved to reports/ (both human-readable .md and artifact files)

## 🔁 Experiments (DVC)

dvc exp run
dvc exp show
dvc exp apply <exp-id>
dvc exp remove <exp-id>

## 💻Tech Stack

Python (3.8+)
scikit-learn /
pandas, numpy, scikit-learn
DVC (pipeline & data versioning)
dvclive (optional, experiment logging)
joblib / cloud storage for artifacts

## 👨‍💻Author

Shayan Ali

