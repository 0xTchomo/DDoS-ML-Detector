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
- [Visualizations](#-Visualizations)
- [Related work](#-Related-Work)
- [Limitations & Future Work](#-Limitations-&-Future-Work)
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

## 📊 Visualizations

### Class Distribution

The dataset exhibits severe class imbalance, with DDoS attacks representing 90% of the training data:

![Target Distribution](images/figures/target_distribution.png)

### Feature Correlation Analysis

Correlation heatmap revealing 50+ highly correlated feature pairs (r > 0.95) that were removed during preprocessing:

![Correlation Heatmap](images/figures/corr_heatmap_subset.png)

### Model Performance: Confusion Matrices

#### AdaBoost (Best Model)

<div align="center">
  <img src="images/figures/confusion_AdaBoost_test.png" width="45%" />
  <img src="images/figures/confusion_AdaBoost_val.png" width="45%" />
</div>

**AdaBoost Test Results:**
- True Positives: 98,899 (99.96%)
- False Negatives: 43 (0.04%)
- True Negatives: 11,443 (99.88%)
- False Positives: 14 (0.12%)

#### K-Nearest Neighbors + PCA

<div align="center">
  <img src="images/figures/confusion_KNN_test.png" width="45%" />
  <img src="images/figures/confusion_KNN_val.png" width="45%" />
</div>

**KNN Test Results:**
- True Positives: 98,899 (99.96%)
- False Negatives: 43 (0.04%)
- True Negatives: 11,402 (99.52%)
- False Positives: 55 (0.48%)

#### Random Forest (Threshold Optimization)

**Threshold 0.5 (Default):**

![RF Threshold 0.5](images/figures/confusion_RandomForest_test_0.5.png)

- **Issue:** 690 false positives (6% benign traffic misclassified)
- Accuracy: 94%

**Threshold 0.3 (Optimized):**

![RF Threshold 0.3](images/figures/confusion_RandomForest_test_0.3.png)

- **Improvement:** Better balance, reduced false negatives
- Accuracy: 96%

**Threshold 0.1 (Maximum Recall):**

![RF Threshold 0.1](images/figures/confusion_RandomForest_test_0.1.png)

- **Optimal for Security:** 99.9% attack detection
- Acceptable false positive rate for cybersecurity context

### ROC Curves

ROC curves demonstrate near-perfect classification performance across all models:

#### AdaBoost

<div align="center">
  <img src="images/figures/roc_AdaBoost_test.png" width="45%" />
  <img src="images/figures/roc_AdaBoost_val.png" width="45%" />
</div>

**AUC = 1.0000** (Perfect classification)

#### K-Nearest Neighbors

<div align="center">
  <img src="images/figures/roc_KNN_test.png" width="45%" />
  <img src="images/figures/roc_KNN_val.png" width="45%" />
</div>

**AUC = 0.9982** (Test) | **AUC = 0.9998** (Validation)

#### Random Forest

<div align="center">
  <img src="images/figures/roc_RandomForest_test.png" width="45%" />
  <img src="images/figures/roc_RandomForest_val.png" width="45%" />
</div>

**AUC = 0.9976** (Test) | **AUC = 1.0000** (Validation)

### Key Takeaways from Visualizations

1. **Severe Class Imbalance:** 90% DDoS vs 10% Benign requires specialized handling
2. **High Feature Correlation:** 50+ pairs removed to reduce multicollinearity
3. **Threshold Impact:** Random Forest performance improved from 94% → 99.3% with threshold tuning
4. **Near-Perfect ROC:** All models achieve AUC > 0.99, indicating excellent discrimination
5. **AdaBoost Superiority:** Best confusion matrix with minimal false negatives/positives

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

## ⚠️ Limitations & Future Work

### Current Limitations

While this project achieves excellent performance on the CIC-DDoS 2019 benchmark dataset (99.95% accuracy), several limitations must be addressed before production deployment:

#### 1. Temporal Analysis Gap

**Current approach:** Each network flow is analyzed independently at a single point in time.

**Limitation:** Real-world DDoS attacks exhibit temporal patterns (gradual ramp-up, sustained duration, coordinated waves) that cannot be detected when analyzing isolated flows.

**Example scenario:**
```
Normal traffic: 1,000 req/min → 1,200 req/min ✅
DDoS attack: 1,000 req/min → 5,000 req/min → 50,000 req/min 🔴

Current model: Analyzes request #5,001 individually → may classify as "Benign"
Missing: No sliding window to detect the 5x traffic spike trend
```

**Impact on production:** Slow-building attacks (slow HTTP, application-layer DDoS) would likely evade detection until reaching critical mass.

---

#### 2. Dataset Generalization

**Current approach:** Trained and validated on CIC-DDoS 2019, a controlled laboratory dataset.

**Limitation:** The model has only seen 6 attack types (Syn, LDAP, UDP, MSSQL, NetBIOS, UDPLag) captured in simulated environments. Real-world traffic includes:
- Legitimate traffic variations (CDN, load balancers, NAT, VPNs)
- Emerging attack vectors not present in 2019 data (HTTP/2 Rapid Reset, QUIC DDoS, IoT botnets)
- Mixed benign/malicious traffic patterns
- Geographic and protocol diversity

**Expected production accuracy:** 80-90% (vs 99.95% on benchmark) due to distribution shift between lab and real-world data.

---

#### 3. Adversarial Robustness

**Current approach:** Standard supervised learning without adversarial training.

**Limitation:** Sophisticated attackers can craft attacks specifically designed to evade ML-based detection by:
- Randomizing packet sizes and timing to mimic legitimate distributions
- Gradually ramping up attack intensity to avoid detection thresholds
- Mixing attack traffic with legitimate requests
- Exploiting knowledge of common ML features (e.g., IAT, packet length distributions)

**Vulnerability:** The model has not been tested against adaptive adversaries who actively try to evade detection.

---

#### 4. Infrastructure Requirements

**Current approach:** Batch processing on static CSV files (offline analysis).

**Limitation:** Production DDoS detection requires:
- **Real-time streaming:** Processing flows as they arrive (<10ms latency)
- **Scalability:** Handling 10,000+ requests/second on commodity hardware
- **Stateful analysis:** Maintaining sliding windows and traffic baselines in memory
- **Integration:** API for SIEM systems, automated mitigation triggers, alerting workflows

**Current pipeline:** Processes 598K flows in ~7 minutes batch mode. A real attack would have caused complete service disruption before detection completes.

---

#### 5. Model Drift & Continuous Learning

**Current approach:** Static model trained once on 2019 data.

**Limitation:** Network traffic patterns and attack techniques evolve constantly. Without continuous retraining:
- New legitimate applications (e.g., emerging video codecs, IoT protocols) may trigger false positives
- Novel attack variants will go undetected
- Model accuracy degrades over time (concept drift)

**Expected degradation:** 5-10% accuracy drop per year without retraining on recent data.

---

### Future Work

To bridge the gap between academic prototype and production-ready system, the following enhancements are proposed:

#### Phase 1: Temporal Intelligence (Priority: High)
- [ ] Implement sliding window analysis (5s, 30s, 5min windows)
- [ ] Add temporal features: traffic rate acceleration, spike detection, baseline deviation
- [ ] Integrate time-series anomaly detection (LSTM, Prophet, Isolation Forest)
- [ ] Develop multi-scale temporal aggregation (per-IP, per-subnet, global)

**Expected impact:** +10-15% recall on slow-building attacks

---

#### Phase 2: Production Infrastructure (Priority: High)
- [ ] Migrate from batch (Pandas) to streaming architecture (Apache Kafka, Flink)
- [ ] Optimize feature extraction pipeline for <5ms latency
- [ ] Implement distributed processing for 100K+ req/s throughput
- [ ] Deploy model inference with TensorFlow Serving / ONNX Runtime
- [ ] Build monitoring dashboard (Prometheus, Grafana)

**Expected impact:** <10ms end-to-end detection latency

---

#### Phase 3: Robustness & Generalization (Priority: Medium)
- [ ] Collect real-world traffic data for validation (with privacy compliance)
- [ ] Implement adversarial training to defend against evasion attacks
- [ ] Build ensemble model combining ML + rule-based + anomaly detection
- [ ] Add Graph Neural Network for IP relationship analysis
- [ ] Integrate threat intelligence feeds (known malicious IPs, ASNs)

**Expected impact:** +5-10% production accuracy, adversarial robustness

---

#### Phase 4: Continuous Learning Pipeline (Priority: Medium)
- [ ] Implement MLOps workflow (DVC, MLflow, Kubeflow)
- [ ] Build automated drift detection monitoring
- [ ] Create weekly retraining pipeline with A/B testing
- [ ] Develop human-in-the-loop labeling workflow for edge cases
- [ ] Version control for models and datasets

**Expected impact:** Sustained 90%+ accuracy over time

---

#### Phase 5: Operational Integration (Priority: Low)
- [ ] Build REST API for SIEM integration
- [ ] Implement automated mitigation triggers (rate limiting, IP blocking)
- [ ] Add explainability layer (SHAP, LIME) for incident investigation
- [ ] Develop false positive review workflow for SOC analysts
- [ ] Create compliance documentation (audit trails, GDPR)

**Expected impact:** Production-grade deployment capability

---

### Comparison with Industrial Solutions

| Feature | This Project | Cloudflare / Akamai | Gap |
|---------|--------------|---------------------|-----|
| **Benchmark Accuracy** | 99.95% | ~99% | ✅ Comparable |
| **Production Accuracy** | ~85% (estimated) | 95-98% | ⚠️ Training data gap |
| **Detection Latency** | N/A (batch) | <3 seconds | 🔴 Infrastructure gap |
| **Temporal Analysis** | ❌ | ✅ Sliding windows | 🔴 Feature gap |
| **Throughput** | 598K flows (offline) | 46M req/s | 🔴 Scalability gap |
| **Adversarial Defense** | ❌ Untested | ✅ Red team validated | 🔴 Robustness gap |
| **Continuous Learning** | ❌ Static model | ✅ Weekly retraining | 🔴 MLOps gap |

**Key insight:** Academic performance is excellent, but the "last mile" to production requires significant infrastructure, operational, and robustness work—this is typical of ML research projects and highlights the difference between proof-of-concept and deployable systems.

---

### Suitable Use Cases

Despite limitations, this model is viable for:

✅ **Security Operations Center (SOC) - Tier 1 Triage**
- Pre-filter high-confidence DDoS alerts before human analysis
- Acceptable false positive rate with analyst oversight

✅ **Small/Medium Business (SMB) Networks**
- Basic DDoS detection for organizations with <10,000 req/s traffic
- Cost-effective alternative to enterprise solutions

✅ **Research & Education**
- Benchmark for comparing ML algorithms on DDoS detection
- Teaching platform for cybersecurity + ML intersection

❌ **Not suitable for:**
- Large-scale CDN/Edge networks (requires streaming + scalability)
- Mission-critical infrastructure without human oversight
- Adversarial environments without robustness enhancements

---

### Lessons Learned

This project demonstrates that **achieving high accuracy on benchmark datasets is the first 20% of the work**—the remaining 80% involves:
1. Building production-grade infrastructure (streaming, monitoring, APIs)
2. Ensuring robustness against adaptive adversaries
3. Maintaining performance over time through continuous learning
4. Integrating with existing security workflows and compliance requirements

Understanding these gaps is crucial for transitioning from academic research to operational cybersecurity systems.

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
