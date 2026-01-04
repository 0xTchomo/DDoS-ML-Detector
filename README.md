# DDoS-ML-Detector

Machine Learning–based system for **DDoS attack detection** using the **CIC-DDoS 2019 dataset**.  
This project focuses on **robust data preprocessing**, **fair model comparison**, and **evaluation under class imbalance**, which are key challenges in real-world intrusion detection systems.

Developed in an academic context (*AI for Threat Detection*), this project reflects a realistic ML workflow for network security applications.

---

## 🎯 Project Goals

- Detect DDoS attacks from network flow statistics
- Build a clean and reproducible ML pipeline
- Compare multiple supervised learning models under the same conditions
- Study the impact of preprocessing and class imbalance on detection performance
- Highlight practical trade-offs relevant to operational IDS environments

---

## 📊 Dataset

- **CIC-DDoS 2019**
- Binary classification:
  - `DDoS`
  - `Benign`
- Highly imbalanced dataset (~90% DDoS, ~10% Benign)

⚠️ Due to this imbalance, accuracy alone is not sufficient to assess model quality.

---

## 🔍 Data Exploration & Cleaning

The dataset was analyzed and cleaned using **statistical and structural criteria only**:

- Removal of constant features
- Removal of identifier-like attributes (e.g., IP addresses, timestamps, flow IDs)
- Correlation analysis to remove highly redundant features (correlation > 0.95)
- Verification of missing values and duplicate samples
- Analysis of class distribution

This step aims to reduce noise and redundancy while preserving general-purpose features suitable for different models.

---

## ⚙️ Preprocessing Pipeline

- Target encoding:
  - `DDoS → 1`
  - `Benign → 0`
- One-hot encoding of categorical protocol features
- Feature scaling using `StandardScaler`
- Stratified train / validation / test splits

Model-specific preprocessing was applied **without altering the core feature selection**:

| Model          | Scaling | PCA |
|---------------|---------|-----|
| Random Forest | Yes     | No  |
| KNN           | Yes     | Yes (95% explained variance) |
| AdaBoost      | Yes     | No  |

---

## 🤖 Models Compared

Three supervised learning models were trained and evaluated independently:

- **Random Forest**
  - Strong baseline for tabular data
  - Robust to noise and non-linear relationships
- **K-Nearest Neighbors (KNN)**
  - Sensitive to high dimensionality
  - Performance improved using PCA
- **AdaBoost**
  - Strong generalization
  - Best overall balance across evaluation metrics

All models were trained on the same cleaned dataset to ensure a fair comparison.

---

## 📈 Evaluation Methodology

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices

Special attention was given to **decision threshold analysis**, highlighting the trade-off between:
- attack detection rate (recall)
- false positive rate on benign traffic

This aspect is critical for real-world intrusion detection systems.

---

## 🧠 Key Observations

- Class imbalance strongly impacts model behavior
- Accuracy alone can be misleading in security datasets
- Threshold calibration significantly affects false positives
- Different ML models respond differently to the same preprocessing
- AdaBoost achieved the most stable performance across metrics

---

## 🚧 Limitations

- Offline, supervised detection only
- No online or streaming traffic analysis
- Dataset is simulated and may not fully represent real network conditions
- No cost-sensitive or probabilistic calibration applied

✅ Why this project matters
This project demonstrates:
a structured ML workflow
awareness of real-world security constraints
careful evaluation beyond raw accuracy
the ability to reason about model trade-offs in intrusion detection

---

## 🔮 Possible Improvements

- Class weighting or resampling strategies
- Precision–Recall AUC optimization
- Online / real-time detection
- Model calibration techniques
- Evaluation on real or mixed traffic datasets

---

## 📂 Repository Structure

```text
├── data/
├── notebooks/
│   └── main.ipynb
├── images/
├── requirements.txt
└── README.md
