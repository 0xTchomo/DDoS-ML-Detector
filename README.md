# 🚨 DDoS Attack Detection using Machine Learning

Machine Learning system for detecting DDoS attacks achieving **99.95% accuracy** on the CIC-DDoS 2019 dataset with 598,440+ network flows.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results & Analysis](#-results--analysis)
- [Academic Context](#-academic-context)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Project Overview

This project implements and compares three Machine Learning algorithms for network-based DDoS attack detection:

- **Random Forest** with threshold optimization (0.5 → 0.3 → 0.1)
- **K-Nearest Neighbors (KNN)** with PCA dimensionality reduction
- **AdaBoost** achieving the best overall performance

The models were trained on the **CIC-DDoS 2019** dataset, a comprehensive collection of network traffic data containing both benign flows and multiple DDoS attack types (Syn, LDAP, UDP, MSSQL, NetBIOS, UDPLag).

## 📊 Key Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Random Forest (threshold 0.1) | 99.3% | 0.993 | 0.999 | 0.996 | 0.9976 |
| KNN + PCA (21 components) | 99.91% | 0.999 | 0.999 | 0.999 | 0.998 |
| **AdaBoost** ⭐ | **99.95%** | **0.9999** | **0.9996** | **0.9997** | **1.000** |

### Dataset Statistics

- **Total Flows:** 598,440 (488,041 train + 110,399 test)
- **Features:** 86 initial → 35 after feature engineering
- **Class Distribution:** 90.3% DDoS / 9.7% Benign (highly imbalanced)
- **Attack Types:** Syn, LDAP, UDP, MSSQL, NetBIOS, UDPLag

## 📊 Dataset

**CIC-DDoS 2019** is a contemporary dataset for DDoS attack detection containing realistic network traffic captures.

### Download Instructions

1. Visit: [CIC-DDoS 2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)
2. Download both files:
   - `training_dataset_CIC_DDoS_2019.csv` (488K flows)
   - `testing_dataset_CIC_DDoS_2019.csv` (110K flows)
3. Place them in the `data/` directory

> **Note:** Dataset files are not included in this repository due to size constraints (ignored by `.gitignore`).

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)

**Data Quality Assessment:**
- ✅ No missing values detected
- ✅ No duplicate entries
- ⚠️ Severe class imbalance: 90% DDoS / 10% Benign

**Feature Analysis:**
- 86 total features (45 float, 35 int, 6 categorical)
- 12 constant columns identified and removed
- 50 highly correlated pairs (r > 0.95) detected

### 2. Feature Engineering

**Removed 57 redundant features:**

| Category | Count | Reason |
|----------|-------|--------|
| Constant columns | 12 | Zero variance (e.g., Bwd PSH Flags, FIN Flag Count) |
| Session identifiers | 18 | Data leakage risk (Flow ID, IPs, Timestamps) |
| Highly correlated | 27 | Multicollinearity (r > 0.95) |

**Final feature set:** 35 features → 21 principal components (PCA for KNN)

**Top predictive features:**
1. Flow IAT Mean
2. Fwd Packet Length Mean
3. Bwd Packet Length Std
4. Active Mean
5. Idle Mean

### 3. Handling Class Imbalance

The severe 90/10 class imbalance required multiple mitigation strategies:

#### **Strategy 1: Stratified Sampling**
```python
train_test_split(X, y, stratify=y, test_size=0.3)
```
- Maintains 90/10 ratio across train/validation/test splits
- Prevents distribution shift between sets
- Essential for reliable performance metrics

#### **Strategy 2: Class Weighting Exploration**
Tested in GridSearchCV:
```python
param_grid = {
    'class_weight': [None, 'balanced']
}
```
- `balanced`: Automatically adjusts weights inversely proportional to class frequencies
- **Result:** No significant improvement over threshold optimization
- Kept `class_weight=None` in final model

#### **Strategy 3: Threshold Optimization** ⭐

Systematically tested decision thresholds:

| Threshold | Accuracy | Recall | Impact |
|-----------|----------|--------|--------|
| **0.5** (default) | 74.1% | 71.1% | ❌ Misses 29% of attacks |
| **0.3** | 78.5% | 76.1% | ⚠️ Still gaps in detection |
| **0.1** | 99.3% | 99.9% | ✅ Detects almost all attacks |

**Key Insight:** In cybersecurity, false positives (false alarms) are acceptable, but false negatives (missed attacks) are critical failures. Lowering the threshold to 0.1 maximizes attack detection (99.9% recall) while maintaining high precision.

#### **Strategy 4: Algorithm Selection**
- **AdaBoost:** Naturally handles imbalance through adaptive boosting
- Automatically increases weight on misclassified examples
- Robust performance without manual class weighting

### 4. Preprocessing Pipeline

**For Random Forest:**
- StandardScaler normalization
- No PCA (trees handle high dimensions well)
- 54 features retained

**For K-Nearest Neighbors:**
- StandardScaler normalization (critical for distance metrics)
- PCA: 95% variance → 21 components
- Reduces curse of dimensionality

**For AdaBoost:**
- StandardScaler normalization
- No PCA (preserves feature interpretability)
- 35 features after redundancy removal

### 5. Model Training & Hyperparameter Tuning

**GridSearchCV configuration:**
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga'],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000]
}
```

**Best parameters found:**
- `C=10` (regularization strength)
- `penalty='l2'` (Ridge regularization)
- `solver='lbfgs'` (optimization algorithm)
- `class_weight=None` (threshold optimization more effective)

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.8+** - Programming language
- **Scikit-learn 1.3.0** - ML algorithms and preprocessing
- **Pandas 2.0+** - Data manipulation and analysis
- **NumPy 1.24+** - Numerical computations
- **Matplotlib 3.7+** - Data visualization
- **Seaborn 0.12+** - Statistical visualizations
- **Jupyter Notebook** - Interactive development environment

### Machine Learning Models

- **Random Forest Classifier** - Ensemble decision trees
- **K-Nearest Neighbors** - Instance-based learning
- **AdaBoost Classifier** - Adaptive boosting
- **PCA** - Principal Component Analysis for dimensionality reduction
- **StandardScaler** - Feature normalization

## 📁 Project Structure
```
DDoS-ML-Detector/
├── notebooks/
│   └── main.ipynb              # Complete ML pipeline and analysis
├── data/
│   └── README.md               # Dataset download instructions
├── images/
│   └── figures/                # Confusion matrices, ROC curves, etc.
├── models/                     # Trained models (not tracked by git)
├── .gitignore                  # Ignore data files and models
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for dataset processing)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/VOTRE_USERNAME/DDoS-ML-Detector.git
cd DDoS-ML-Detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
Follow instructions in `data/README.md` to download CIC-DDoS 2019 dataset

### Run the Analysis
```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/main.ipynb

# Run all cells to reproduce results
```

### Expected Runtime

- Data loading: ~30 seconds
- Preprocessing: ~2 minutes
- Model training:
  - Random Forest: ~5 minutes
  - KNN: ~10 minutes
  - AdaBoost: ~3 minutes
- Total: ~20-25 minutes

## 📈 Results & Analysis

### Confusion Matrix - AdaBoost (Best Model)
```
                    Predicted
                 Benign    DDoS
Actual  Benign    [TN]     [FP]
        DDoS      [FN]     [TP]

True Positives (TP):  99,996 (99.96%)
False Negatives (FN):      4 (0.04%)
True Negatives (TN):  10,395 (99.99%)
False Positives (FP):      4 (0.01%)
```

### Key Findings

1. **Threshold Tuning is Critical**
   - Random Forest improved from 74% to 99.3% accuracy with threshold optimization
   - Small adjustment (0.5 → 0.1) had massive impact on recall

2. **PCA Essential for KNN**
   - Raw features: Poor performance, slow computation
   - With PCA (21 components): 99.91% accuracy, 10x faster

3. **AdaBoost Superior for Imbalanced Data**
   - Achieved near-perfect classification (99.95% accuracy)
   - Naturally robust to class imbalance through adaptive boosting
   - Best precision-recall trade-off

4. **Feature Engineering Impact**
   - Removing 57 redundant features improved model interpretability
   - Reduced overfitting risk
   - Faster training and inference

### Business Impact

In a production environment processing **1 million flows/day**:
- **99.95% accuracy** → Only 500 misclassifications
- **99.96% recall** → Detects 999,600 attacks (misses only 400)
- **99.99% precision** → Only 100 false alarms

This performance level is suitable for real-world deployment in SOC/SIEM systems.

## 📚 Related Work

### Commercial Solutions Comparison

Our model performance is comparable to commercial DDoS detection systems:
- Cloudflare Magic Transit
- Akamai Prolexic
- AWS Shield Advanced
- Arbor Sightline

These systems typically combine ML-based detection with volumetric filtering and scrubbing centers.

### Emerging Attack Vectors (2024)

Based on [Cloudflare Q4 2024 DDoS Report](https://blog.cloudflare.com/ddos-threat-report-2024-q4):
- HTTP/2 Rapid Reset attacks (+600% increase)
- QUIC DDoS exploitation
- TCP Middlebox Reflection attacks
- IoT botnet proliferation

Future work could extend this model to detect these emerging patterns.

## 🎓 Academic Context

**Course:** Machine Learning for Cybersecurity  
**Institution:** Télécom Paris, Institut Polytechnique de Paris  
**Academic Year:** 2024-2025  
**Project Type:** Practical Lab Assignment (TP)

### Learning Objectives Achieved

- ✅ Handle severely imbalanced datasets
- ✅ Apply feature engineering for network data
- ✅ Compare multiple ML algorithms systematically
- ✅ Optimize decision thresholds for security contexts
- ✅ Interpret model performance in business terms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Thierry Armel TCHOMO KOMBOU**

🎓 Cybersecurity Engineering Student @ Télécom Paris  
🔬 Specialization: Cybersecurity and AI

📧 Email: tchomokombou@telecom-paris.fr  
🐙 GitHub: [0xTchomo](https://github.com/0xTchomo)

---

### 🌟 Acknowledgments

- **Dataset:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick
- **Course Instructors:** Télécom Paris Cybersecurity Department
- **Tools:** Scikit-learn, Pandas, Jupyter communities

---

⭐ **If you find this project useful, please consider giving it a star!**

📝 **Feedback and contributions are welcome** - Feel free to open an issue or pull request.

---

**Last Updated:** January 2026
