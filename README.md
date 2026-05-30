<div align="center">

<img src="https://img.icons8.com/ios-filled/100/e63946/heart-with-pulse.png" width="80" alt="Heart Icon"/>

# ❤️ Heart Disease Risk Predictor with Explainable AI

### A Clinical Decision-Support Dashboard powered by XGBoost, SHAP, and LIME

[![Streamlit] <img width="1859" height="972" alt="Image" src="https://github.com/user-attachments/assets/7042c8cf-927d-4a8a-a337-e487fc9e806c" />

<br/>

> **Predict heart disease risk with full transparency.**  
> This project tackles the "black-box" problem in medical AI by pairing a high-accuracy XGBoost classifier with SHAP and LIME explainability layers, deployed as an interactive Streamlit web application.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Explainability (XAI)](#-explainability-xai)
- [Output Plots](#-output-plots)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Cardiovascular disease is the leading cause of death globally. Early, accurate, and *explainable* risk prediction can dramatically improve clinical outcomes. This project addresses two core challenges in medical AI:

1. **Accuracy** — Trained on the **UCI Heart Disease dataset** (920 patients across 4 hospital sources), the XGBoost model achieves **84.78% accuracy** and a **0.9005 ROC-AUC**.
2. **Transparency** — Clinicians and patients cannot act on a prediction they don't understand. **SHAP** explains *global* feature importance across all patients; **LIME** explains *individual* predictions locally, model-agnostically.

The result is a Streamlit dashboard where a clinician inputs patient vitals and immediately receives a risk score alongside a plain-language breakdown of *which features drove the prediction and by how much*.

---

## 🚀 Live Demo

> 🔗 **Deploy your own:** Follow the [Installation](#-installation) steps and run `streamlit run app.py`.  
> Or deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) — free, one-click, no server needed.

---

## 🏗️ Project Architecture

```
Patient Vitals (13 clinical features)
           │
           ▼
  ┌─────────────────────┐
  │  Preprocessing       │  Median imputation · One-hot encoding
  │  + StandardScaler    │  · Train/Test split (80/20, stratified)
  └────────┬────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
 Random         XGBoost  ◄── Best Model
 Forest         Classifier    (AUC 0.9005)
  (AUC 0.9250)
     │            │
     └─────┬──────┘
           ▼
  ┌─────────────────────┐
  │   XAI Layer          │
  │  ├── SHAP (global)   │  Feature ranking across all patients
  │  ├── SHAP (local)    │  Per-patient contribution breakdown
  │  └── LIME (local)    │  Model-agnostic linear approximation
  └────────┬────────────┘
           ▼
  ┌─────────────────────┐
  │  Streamlit App       │  Risk score · Probability · Visual charts
  └─────────────────────┘
```

---

## 📊 Dataset

**UCI Heart Disease Dataset** — one of the most widely-cited cardiovascular datasets in machine learning research.

| Property | Detail |
|---|---|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) / [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| **Rows** | 920 patients |
| **Features** | 13 clinical input features + 1 target |
| **Hospitals** | Cleveland, Hungary, Switzerland, VA Long Beach |
| **Target** | `num`: 0 = No Disease, 1–4 = Increasing Severity → **Binarised to 0/1** |

### Feature Reference Table

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age in years |
| `sex` | Categorical | Biological sex (Male / Female) |
| `cp` | Categorical | Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic) |
| `trestbps` | Numeric | Resting blood pressure (mm Hg) |
| `chol` | Numeric | Serum cholesterol (mg/dL) |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dL (True / False) |
| `restecg` | Categorical | Resting ECG results (normal, ST-T abnormality, LV hypertrophy) |
| `thalch` | Numeric | Maximum heart rate achieved |
| `exang` | Binary | Exercise-induced angina (True / False) |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of peak exercise ST segment (upsloping, flat, downsloping) |
| `ca` | Numeric | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Categorical | Thalassemia type (normal, fixed defect, reversable defect) |
| **`num`** | **Target** | **Heart disease diagnosis (0 = healthy, 1 = disease)** |

---

## ✨ Features

- 🩺 **Interactive Patient Input Panel** — 13 clinical sliders and dropdowns in the sidebar
- 🎯 **Instant Risk Prediction** — Probability score with High / Low risk classification
- 🔴 **SHAP Global Summary** — Ranked feature importance across the entire dataset
- 🟢 **SHAP Local Explanation** — Per-patient bar chart showing which features pushed the prediction
- 🧪 **LIME Local Explanation** — Independent model-agnostic validation of SHAP findings
- 📈 **ROC Curve Comparison** — Side-by-side Random Forest vs XGBoost performance
- 🔬 **Confusion Matrix** — Test-set breakdown of TP, TN, FP, FN
- 🏥 **Multi-hospital Dataset** — Trained on 4 global hospital sources for generalisability
- 📋 **Clinical Pathway Guidelines** — Risk-tiered action recommendations built into the UI

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Recommended) Virtual environment

### Step 1 — Clone the Repository

```bash
git clone https://github.com/ChaitanyaSkr5/heart-disease-predictor-xai.git
cd heart-disease-predictor-xai
```

### Step 2 — Create & Activate a Virtual Environment *(recommended)*

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Verify Dataset

Ensure `heart_disease_uci.csv` is present in the project root. The dataset is included in this repository. If you need to re-download it:
- [Kaggle Source](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- [UCI Source](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Step 5 — Train the Model *(if `.pkl` files are absent)*

```bash
jupyter notebook Heart_Disease_Predictor_XAI_model_training.ipynb
# Run all cells — this saves best_xgb_model.pkl and scaler.pkl
```

---

## 🖥️ Usage

### Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### How to Use

1. **Fill in the Patient Profile** using the sidebar sliders and dropdowns (age, BP, cholesterol, etc.)
2. **View the Diagnostic Assessment** — risk label (High / Low), probability score, and clinical path guidelines
3. **Explore SHAP Explanations** — see which features contributed positively or negatively to the prediction
4. **Review LIME Explanation** — independent local explanation to cross-validate the SHAP output
5. **Check Model Performance** — ROC curves, confusion matrix, and metrics comparison tab

### Run the Jupyter Notebook

```bash
jupyter notebook Heart_Disease_Predictor_XAI_model_training.ipynb
```

The notebook walks through every step: EDA → Preprocessing → Model Training → Evaluation → SHAP → LIME → Hyperparameter Tuning.

---

## 📈 Model Performance

Evaluated on a held-out test set of **184 patients** (20% stratified split, `random_state=42`).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 83.15% | 0.83 | 0.83 | 0.83 | **0.9250** |
| **XGBoost** ⭐ | **84.78%** | **0.85** | **0.85** | **0.84** | **0.9005** |

> ⭐ XGBoost is saved as `best_xgb_model.pkl` and used in the Streamlit app.

### Cross-Validation (5-Fold Stratified)

| Model | CV ROC-AUC (Mean) |
|---|---|
| Random Forest | 0.8793 |
| XGBoost | 0.8655 |

---

## 🔍 Explainability (XAI)

This project uses **two independent XAI methods** that are verified to agree on the top features — increasing clinical confidence in the model's reasoning.

### SHAP — SHapley Additive exPlanations

Based on cooperative game theory, SHAP fairly distributes the prediction contribution among all features.

- **Global summary plot** — dot plot ranking all 13 features by mean absolute SHAP value across 184 test patients. Color shows whether high (red) or low (blue) feature values push toward disease.
- **Local bar chart** — for the current patient, each bar shows how much that feature pushed the prediction above or below the base rate.

**Top features identified by SHAP:**

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `cp` (Chest Pain Type) | Asymptomatic chest pain strongly correlates with disease |
| 2 | `thal` (Thalassemia) | Reversible defect is a strong positive predictor |
| 3 | `thalch` (Max Heart Rate) | Lower max HR associated with higher disease risk |
| 4 | `ca` (Vessels Coloured) | More vessels blocked → higher risk |
| 5 | `oldpeak` (ST Depression) | Higher ST depression → higher risk |

### LIME — Local Interpretable Model-agnostic Explanations

LIME perturbs the input space around a single patient and fits a local linear model to approximate the black-box decision boundary. It is **model-agnostic** — it validates SHAP without any access to SHAP's internals.

**Agreement:** LIME independently ranks `cp`, `thal`, and `thalch` as the top 3 features for the majority of test patients, confirming SHAP's findings.

---

## 🖼️ Output Plots

All plots are saved to the `model_output_plots/` directory after running the notebook.

| File | Description |
|---|---|
| `01_feature_distributions.png` | 2×4 histogram grid — healthy vs disease distributions for 8 key features |
| `02_correlation_matrix.png` | Full heatmap of linear correlations among all numeric predictors |
| `03_roc_curves.png` | Side-by-side ROC curves for Random Forest and XGBoost with AUC scores |
| `04_shap_global_summary.png` | SHAP dot summary plot — global feature importance ranked top-to-bottom |
| `05_lime_local_patient_0.png` | LIME bar chart — local explanation for Patient Index #0 |

---

## 📁 Project Structure

```
heart-disease-predictor-xai/
│
├── app.py                                        # Streamlit web application
├── Heart_Disease_Predictor_XAI_model_training.ipynb  # Full training notebook
├── heart_disease_uci.csv                         # UCI Heart Disease dataset (920 rows)
├── best_xgb_model.pkl                            # Trained XGBoost model
├── scaler.pkl                                    # Fitted StandardScaler
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
│
└── model_output_plots/
    ├── 01_feature_distributions.png
    ├── 02_correlation_matrix.png
    ├── 03_roc_curves.png
    ├── 04_shap_global_summary.png
    ├── 05_lime_local_patient_0.png
    ├── plot_index_descriptions.txt
    └── web app UI.png
```

---

## 🛠️ Tech Stack

| Category | Tool / Library | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Core language |
| ML Models | scikit-learn | 1.4+ | Random Forest, preprocessing, metrics |
| ML Models | XGBoost | 2.0+ | Gradient boosted tree classifier |
| XAI | SHAP | 0.45+ | Global + local Shapley explanations |
| XAI | LIME | 0.2+ | Model-agnostic local explanations |
| Web App | Streamlit | 1.32+ | Interactive dashboard |
| Data | Pandas | 2.1+ | Data manipulation |
| Data | NumPy | 1.26+ | Numerical computation |
| Visualisation | Matplotlib | 3.8+ | Plots and charts |
| Visualisation | Seaborn | 0.13+ | Statistical visualisations |
| Serialisation | joblib | 1.3+ | Model saving and loading |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes with a descriptive message:
   ```bash
   git commit -m "feat: add multi-class severity prediction support"
   ```
4. **Push** to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** — describe what you changed and why

### Good First Issues

- [ ] Add SHAP waterfall plot for individual patients
- [ ] Extend to multi-class prediction (severity levels 0–4)
- [ ] Add patient history comparison across sessions
- [ ] Integrate hyperparameter tuning UI via GridSearchCV
- [ ] Deploy to Streamlit Community Cloud with public URL

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Add docstrings to all functions
- Include comments explaining XAI-specific logic

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

```
MIT License

Copyright (c) 2026 Chaitanya Skr5

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👤 Author

<div align="center">

| | |
|---|---|
| **Name** | Chaitanya Skr5 |
| **Email** | [chaitanyasurkar54@gmail.com](mailto:chaitanyasurkar54@gmail.com) |
| **GitHub** | [@ChaitanyaSkr5](https://github.com/ChaitanyaSkr5) |
| **Department** | Artificial Intelligence |
| **Institution** | GH Raisoni College of Engineering, Nagpur |

</div>

---

## 🙏 Acknowledgements

- **UCI Machine Learning Repository** — for the original Heart Disease dataset curated from Cleveland, Hungarian, Swiss, and VA Long Beach hospital records
- **Scott Lundberg et al.** — for the [SHAP library](https://github.com/slundberg/shap) and the foundational paper *"A Unified Approach to Interpreting Model Predictions"*
- **Marco Tulio Ribeiro et al.** — for the [LIME library](https://github.com/marcotcr/lime) and the paper *"Why Should I Trust You?: Explaining the Predictions of Any Classifier"*
- **XGBoost Team** — for the [XGBoost](https://github.com/dmlc/xgboost) gradient boosting framework
- **Streamlit** — for making it dead simple to turn ML models into interactive web apps

---

## ⚠️ Disclaimer

> This tool is built for **educational and portfolio purposes only**.  
> It is **not** a certified medical device and should **not** be used as a substitute for professional clinical diagnosis or advice. Always consult a qualified healthcare provider for medical decisions.

---

<div align="center">

Made with ❤️ by **Chaitanya Skr5**  
⭐ If this project helped you, please give it a star on GitHub!

</div>
