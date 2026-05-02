# 🛡️ ML Guardian

Automated model quality gates for MLflow. When a new model version is trained, ML Guardian compares it against the current production model, evaluates configurable quality gates, and either auto-promotes it or blocks deployment with a detailed report.

Built to work natively with [`ml-production-template`](https://github.com/Eklavya20/ml-production-template) and [`diagnost`](https://github.com/Eklavya20/diagnost).

[![CI](https://github.com/Eklavya20/ml-guardian/actions/workflows/ml-guardian.yml/badge.svg)](https://github.com/Eklavya20/ml-guardian/actions/workflows/ml-guardian.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/ml-guardian/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🚀 Why ML Guardian

Retraining a model and manually promoting it in the MLflow UI is error-prone and undocumented. ML Guardian replaces that manual step with an automated, auditable gate that:

- Compares accuracy, F1, ROC-AUC, precision, recall against the live production model  
- Flags metric regressions beyond configurable thresholds  
- Checks calibration quality via Expected Calibration Error (ECE)  
- Detects input feature drift between training runs  
- Auto-promotes the candidate if all gates pass, or blocks and exits with a non-zero code  

---

## 🏗️ Architecture

```
New training run completes
        │
        ▼
MLflow Model Registry
(candidate in None/Staging)
        │
        ▼
┌─────────────────────────┐
│      ML Guardian        │
│                         │
│  loader     → fetch     │
│  comparator → metrics   │
│  gates      → pass/fail │
│  reporter   → markdown  │
│  promoter   → promote   │
└──────────┬──────────────┘
           │
     ┌─────┴──────┐
     │            │
   PASSED       FAILED
     │            │
 Promote      Block + exit 1
 to Prod      report saved
```

---

## ⚡ Quickstart

### Prerequisites

- Python 3.10+
- MLflow tracking server running  
- Model registered in MLflow  
- `X_test.csv` and `y_test.csv` logged as artifacts  

---

### Install

```bash
git clone https://github.com/Eklavya20/ml-guardian.git
cd ml-guardian
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

---

### Run

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_NAME=telco_churn
export AUTO_PROMOTE=true

python -m src.main
```

**PowerShell:**
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:MODEL_NAME="telco_churn"
$env:AUTO_PROMOTE="true"

python -m src.main
```

---

## ⚙️ Configuration

All thresholds are controlled via environment variables:

| Variable | Default | Description |
|----------|--------|------------|
| MLFLOW_TRACKING_URI | http://localhost:5000 | MLflow server URI |
| MODEL_NAME | telco_churn | Registered model name |
| AUTO_PROMOTE | true | Auto-promote if gates pass |
| MIN_ACCURACY | 0.70 | Minimum candidate accuracy |
| MIN_F1 | 0.65 | Minimum candidate F1 |
| MIN_ROC_AUC | 0.70 | Minimum candidate ROC-AUC |
| MAX_ACCURACY_DROP | 0.02 | Max allowed accuracy regression |
| MAX_F1_DROP | 0.03 | Max allowed F1 regression |
| MAX_ROC_AUC_DROP | 0.02 | Max allowed ROC-AUC regression |
| MAX_ECE | 0.15 | Max Expected Calibration Error |
| MAX_DRIFTED_FEATURES | 3 | Max allowed drifted features |

---

## 📊 Example Report

```
🛡️ ML Guardian Report
Model: telco_churn
Candidate run: cdead1b3721344caaa749c1ed819840f
Production run: e3b59a5b5a09490086938177ab9e9efd
Generated: 2026-04-18 12:13 UTC

Status: ✅ PASSED — Auto-promotion approved

📊 Metric Comparison
Metric       Candidate   Production   Delta
accuracy     0.7686      0.7686       0.0000
f1           0.7767      0.7767       0.0000
roc_auc      0.8358      0.8358       0.0000
precision    0.7936      0.7936       0.0000
recall       0.7686      0.7686       0.0000

🎯 Calibration
ECE: 0.0933 → Moderately calibrated

🌊 Drift Detection
Drifted features: 0 / 19
```

---


## 📊 Example Reports

### ✅ Passed — Auto-promotion approved

\```
🛡️ ML Guardian Report
Model: telco_churn
Candidate run: cdead1b3721344caaa749c1ed819840f
Production run: e3b59a5b5a09490086938177ab9e9efd
Generated: 2026-04-18 12:13 UTC

Status: ✅ PASSED — Auto-promotion approved

📊 Metric Comparison
Metric       Candidate   Production   Delta
accuracy     0.7686      0.7686       0.0000
f1           0.7767      0.7767       0.0000
roc_auc      0.8358      0.8358       0.0000

🎯 Calibration
ECE: 0.0933 → Moderately calibrated

🌊 Drift Detection
Drifted features: 0 / 19
\```

---

### ❌ Failed — Promotion blocked

Triggered by a model trained on 10% of data with `n_estimators=1`.
All three regression gates fired. Production model was protected.

# 🛡️ ML Guardian Report
**Model:** `telco_churn`
**Candidate run:** `162b638e3db54003a689de5cc6bf99c4`
**Production run:** `cdead1b3721344caaa749c1ed819840f`
**Generated:** 2026-05-02 20:55 UTC

## Status: ❌ FAILED — Promotion blocked

### ❌ Gate Failures
- Accuracy regressed by 0.1386 (max allowed drop: 0.02)
- F1 regressed by 0.1379 (max allowed drop: 0.03)
- ROC-AUC regressed by 0.2129 (max allowed drop: 0.02)

### 📊 Metric Comparison
| Metric | Candidate | Production | Delta |
|--------|-----------|------------|-------|
| accuracy | 0.7215 | 0.8601 | 🔴 -0.1386 |
| f1 | 0.7277 | 0.8656 | 🔴 -0.1379 |
| roc_auc | 0.7309 | 0.9438 | 🔴 -0.2129 |
| precision | 0.7365 | 0.8846 | 🔴 -0.1481 |
| recall | 0.7215 | 0.8601 | 🔴 -0.1386 |

### 🎯 Calibration
| | Candidate | Production |
|---|-----------|------------|
| ECE | 0.0464 | 0.1005 |
| Verdict | Well calibrated | Poorly calibrated |

### 🌊 Drift Detection
**Drifted features:** 0 / 19

---

## 📦 Logging Test Artifacts

ML Guardian requires test datasets logged as MLflow artifacts:

```python
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

mlflow.log_artifact("X_test.csv")
mlflow.log_artifact("y_test.csv")

os.remove("X_test.csv")
os.remove("y_test.csv")
```

---

## 🔁 GitHub Actions (CI Setup)

### Add Repository Secrets

- `MLFLOW_TRACKING_URI`
- `MODEL_NAME`

---

### CI Workflow

Create or update:

```
.github/workflows/ml-guardian.yml
```

```yaml
name: ML Guardian CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=term-missing

      - name: Upload coverage artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: .coverage
```

---

## 🧪 Run Tests Locally

```bash
pytest tests/ -v --cov=src
```

---

## 📁 Project Structure

```
ml-guardian/
├── .github/workflows/
│   └── ml-guardian.yml
├── src/
│   ├── loader.py
│   ├── comparator.py
│   ├── gates.py
│   ├── reporter.py
│   ├── promoter.py
│   └── main.py
├── tests/
│   ├── test_comparator.py
│   ├── test_gates.py
│   └── test_reporter.py
├── reports/
└── pyproject.toml
```

---

## 🔗 Related Projects

- [`diagnost`](https://github.com/Eklavya20/diagnost) — model diagnostics engine  
- [`ml-production-template`](https://github.com/Eklavya20/ml-production-template) — full MLOps stack  

---

## 📜 License

MIT

---