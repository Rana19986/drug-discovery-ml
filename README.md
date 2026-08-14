# 🧬 Drug Discovery ML Pipeline

An ML-powered molecular property prediction and virtual screening pipeline built with RDKit, XGBoost, and FastAPI.

## 🚀 Features

- Molecular solubility prediction
- RDKit molecular descriptors
- Morgan fingerprints
- Lipinski Rule of Five analysis
- Molecular similarity search
- Virtual screening
- SHAP model explainability
- FastAPI REST API

## 📊 Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost + Descriptors + Morgan | 0.4832 | 0.6747 | 0.9037 |

Scaffold split validation:

**R²: 0.8617**

## 🛠️ Tech Stack

- Python
- RDKit
- XGBoost
- Scikit-learn
- NumPy
- FastAPI
- SHAP

## 🔬 Pipeline

SMILES → RDKit Descriptors + Morgan Fingerprints → XGBoost → Solubility Prediction

## ⚠️ Disclaimer

This project is intended for research and educational purposes. Predictions should not be interpreted as clinical or medical recommendations. 
