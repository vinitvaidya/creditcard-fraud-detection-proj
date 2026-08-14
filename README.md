# Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB5E28)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/Deployed%20on-AWS-232F3E?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![CI/CD](https://github.com/<vinitvaidya>/creditcard-fraud-detection-proj/actions/workflows/main.yaml/badge.svg)](https://github.com/<vinitvaidya>/creditcard-fraud-detection-proj/actions/workflows/main.yaml)
[![License](https://img.shields.io/github/license/<vinitvaidya>/creditcard-fraud-detection-proj)](LICENSE)

An end-to-end machine learning system that flags potentially fraudulent credit card transactions in real time. It covers the full lifecycle — data ingestion, validation, transformation, model training, experiment tracking, and deployment — behind a FastAPI service that can be queried via a JSON API or a browser-based console.

---

## 1. Project Overview

Credit card fraud is rare but expensive when missed. This project trains an **XGBoost classifier** to distinguish fraudulent transactions from legitimate ones on a real-world, highly imbalanced dataset, handles the class imbalance with **SMOTE**, tracks every experiment with **MLflow**, and ships the final model behind a **FastAPI** service that's containerized and deployed to **AWS** through a GitHub Actions CI/CD pipeline.

The goal is not just a notebook that scores well — it's a reproducible pipeline that can be retrained, evaluated, and redeployed with a single `git push`.

---

## 2. Dataset

The dataset contains transactions made by European cardholders' credit cards over two days in September 2013.

- **284,807** total transactions, of which only **492 are fraudulent** — the positive (fraud) class makes up just **0.172%** of the data.
- All input features except two have been transformed via **PCA** for confidentiality — `V1` through `V28` are the resulting principal components, with no access to the original features or further background on what they represent.
- **`Time`** — seconds elapsed between each transaction and the first transaction in the dataset. Not PCA-transformed.
- **`Amount`** — the transaction amount. Not PCA-transformed; usable for example/cost-sensitive learning.
- **`Class`** — the target variable: `1` for fraud, `0` otherwise.

Because of the extreme class imbalance, **accuracy is not a meaningful evaluation metric** here — a model predicting "not fraud" every time would already score ~99.8%. The dataset's own documentation recommends evaluating with the **Area Under the Precision-Recall Curve (AUPRC)** rather than relying on confusion-matrix-derived accuracy, which is why this project also tracks precision, recall, F1, and ROC-AUC (see [Model Performance](#12-model-performance)) rather than accuracy alone.

Source: [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) on Kaggle, downloaded programmatically via `kagglehub`.

---

## 3. Features

- **Config-driven ML pipeline** — data ingestion, validation, transformation, training, and evaluation are all controlled through YAML config files (`config.yaml`, `params.yaml`, `schema.yaml`), not hardcoded values.
- **Automated dataset ingestion** from Kaggle via `kagglehub`, with a custom cache location inside the project instead of the default system cache.
- **Schema validation** — every incoming dataset is checked against `schema.yaml` before training proceeds, so a malformed dataset fails fast instead of silently corrupting a training run.
- **Class-imbalance handling** with SMOTE, applied only to training data to avoid data leakage into evaluation.
- **Experiment tracking with MLflow**, hosted remotely on DagsHub — every run logs parameters, metrics (precision, recall, F1, ROC-AUC), a full classification report, and a confusion matrix as artifacts.
- **Two prediction interfaces**:
  - A JSON REST API (`/predict`) for programmatic access, documented via Swagger UI.
  - A browser-based console (`/`, `/predict-form`) for manual transaction analysis, rendered with Jinja2 templates.
- **One-click retraining** via a `/train` endpoint that re-runs the full pipeline.
- **Containerized deployment** — the entire service ships as a Docker image.
- **Automated CI/CD** — every push to `main` builds the image, pushes it to AWS ECR, and redeploys it to an EC2 instance via a self-hosted GitHub Actions runner.

---

## 4. Tech Stack

| Technology | Role in this project |
|---|---|
| **Python 3.11** | Core language for the pipeline and API. |
| **XGBoost** | Gradient-boosted tree classifier — chosen for strong performance on tabular, imbalanced data and built-in handling of feature interactions without heavy preprocessing. |
| **scikit-learn / imbalanced-learn (SMOTE)** | Evaluation metrics and synthetic oversampling of the minority (fraud) class during training. |
| **kagglehub** | Programmatic dataset download from Kaggle, replacing manual CSV downloads and keeping data acquisition scriptable and reproducible. |
| **MLflow + DagsHub** | Experiment tracking and artifact storage. DagsHub hosts the MLflow-compatible tracking server remotely, so experiment history isn't tied to one machine. |
| **FastAPI** | Serves both the JSON prediction API and the HTML console. Chosen over Flask for built-in request validation (via Pydantic), automatic OpenAPI/Swagger docs, and async support. |
| **Pydantic** | Defines and validates the shape of incoming prediction requests (`UserInput` schema). |
| **Jinja2** | Server-side templating for the HTML console (`index.html`, `result.html`). |
| **joblib** | Model persistence — serializes the trained XGBoost pipeline for reuse at inference time. |
| **Docker** | Packages the application and its dependencies into a single deployable image. |
| **AWS ECR** | Stores built Docker images. |
| **AWS EC2** | Hosts the running container that serves predictions. |
| **GitHub Actions (self-hosted runner)** | Automates build → push → deploy on every push to `main`. |
| **python-dotenv** | Loads local secrets (Kaggle credentials, cache paths) from a `.env` file during local development. |

---

## 5. Architecture

```
                         ┌─────────────────────────┐
                         │        Kaggle            │
                         │  (creditcardfraud data)  │
                         └────────────┬─────────────┘
                                      │ kagglehub
                                      ▼
┌───────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                          │
│                                                                     │
│  Data Ingestion → Data Validation → Data Transformation (SMOTE)    │
│         │                │                     │                   │
│         ▼                ▼                     ▼                   │
│   artifacts/data    schema.yaml check     train.csv / test.csv     │
│                                                     │               │
│                                                     ▼               │
│                                          Model Trainer (XGBoost)    │
│                                                     │               │
│                                                     ▼               │
│                                          Model Evaluation           │
│                                       (metrics + confusion matrix)  │
└───────────────────────────────┬───────────────────┬─────────────────┘
                                 │                   │
                                 ▼                   ▼
                     artifacts/model_trainer/    MLflow (via DagsHub)
                        model.joblib                remote tracking
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │      FastAPI App         │
                    │  ┌───────────────────┐  │
                    │  │ /predict (JSON)    │  │
                    │  │ /predict-form(HTML)│  │
                    │  │ /train             │  │
                    │  └───────────────────┘  │
                    └────────────┬─────────────┘
                                 │  Docker image
                                 ▼
                          ┌─────────────┐
                          │   AWS ECR    │
                          └──────┬──────┘
                                 │ pull on deploy
                                 ▼
                          ┌─────────────┐
                          │   AWS EC2    │──▶  End user (browser / API client)
                          └─────────────┘
                                 ▲
                                 │ self-hosted runner
                    ┌────────────────────────┐
                    │   GitHub Actions CI/CD   │
                    │  build → push → deploy   │
                    └────────────────────────┘
```

---

## 6. Project Structure

```
creditcard-fraud-detection-proj/
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD: build image → push to ECR → deploy to EC2
├── artifacts/                      # Generated at runtime by the pipeline
│   ├── data_ingestion/
│   │   └── datasets/mlg-ulb/creditcardfraud/versions/3/
│   │       └── creditcard.csv      # Raw dataset, downloaded via kagglehub
│   ├── data_transformation/
│   │   ├── train.csv                # SMOTE-balanced training split
│   │   └── test.csv                 # Untouched evaluation split
│   ├── data_validation/
│   │   └── status.txt                # Schema validation result
│   ├── model_trainer/
│   │   └── model.joblib              # Trained XGBoost model
│   └── model_evaluation/
│       ├── metrics.json               # accuracy, precision, recall, f1, roc_auc
│       ├── classification_report.txt  # Full sklearn classification report
│       └── confusion_matrix.json      # Confusion matrix values
├── config/
│   └── config.yaml                 # Paths for artifacts, data, and model files
├── research/                        # Exploratory notebooks — one per pipeline stage
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformation.ipynb
│   ├── 04_model_trainer.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── trials.ipynb
├── schemas/
│   └── user_input.py                # Pydantic model validating API request bodies
├── src/cred_card_proj/
│   ├── components/                  # Core pipeline logic, one module per stage
│   │   ├── data_ingestion.py         # Downloads dataset via kagglehub
│   │   ├── data_validation.py        # Validates columns against schema.yaml
│   │   ├── data_transformation.py    # Train/test split + SMOTE
│   │   ├── model_trainer.py          # Trains the XGBoost classifier
│   │   └── model_evaluation.py       # Computes metrics, logs to MLflow
│   ├── config/
│   │   └── configuration.py          # Reads YAML configs into typed entities
│   ├── constants/                    # Shared constant values (file paths, etc.)
│   ├── entity/
│   │   └── config_entity.py          # Dataclasses defining each stage's config shape
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── prediction.py             # PredictionPipeline used by the API
│   └── utils/
│       └── common.py                 # Shared helpers (read_yaml, save_json, etc.)
├── templates/
│   ├── index.html                    # Transaction input console (form)
│   └── result.html                   # Verdict / prediction result page
├── logs/
│   └── running_logs.log              # Pipeline run logs
├── schema.yaml                       # Expected columns + dtypes for the dataset
├── params.yaml                       # Model hyperparameters
├── app.py                            # FastAPI entrypoint (API + HTML routes)
├── main.py                           # Runs the full training pipeline end-to-end
├── template.py                       # Scaffolds the project's folder/file structure
├── setup.py                          # Makes src/cred_card_proj installable as a package
├── Dockerfile                        # Container definition
├── requirements.txt
└── .env                              # Local secrets (not committed) — Kaggle creds, cache path
```

*(Virtual environment, `__pycache__`, and build metadata folders are omitted above for readability.)*

---

## 7. Installation & Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized runs)
- A Kaggle account with an API token
- A DagsHub account with an MLflow-enabled repo (for experiment tracking)
- An AWS account with ECR + EC2 access (only required for deployment, not local development)

### 1. Clone the repository

```bash
git clone https://github.com/<vinitvaidya>/creditcard-fraud-detection-proj.git
cd creditcard-fraud-detection-proj
```

### 2. Create and activate a virtual environment

```bash
python -m venv fraud_detection_env
# Windows
fraud_detection_env\Scripts\activate
# macOS/Linux
source fraud_detection_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
KAGGLEHUB_CACHE=./artifacts/data_ingestion

MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 5. Run the training pipeline

```bash
python main.py
```

This runs all five stages in sequence — ingestion, validation, transformation, training, evaluation — and logs the run to MLflow. The trained model is saved to `artifacts/model_trainer/model.joblib`.

### 6. Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Running with Docker instead

```bash
docker build -t fraud-detection .
docker run -d -p 8080:8080 \
  -e KAGGLE_USERNAME=your_kaggle_username \
  -e KAGGLE_KEY=your_kaggle_api_key \
  -e KAGGLEHUB_CACHE=/app/artifacts/data_ingestion \
  fraud-detection
```

---

## 8. Usage

1. Visit `http://localhost:8080/` to open the transaction console.
2. Enter the 30 feature values for a transaction (`Time`, `V1`–`V28`, `Amount`).
3. Submit the form — the app runs the transaction through the trained model and shows a verdict: **Transaction Legitimate** or **Fraud Detected**.
4. To retrain the model with fresh data, hit `GET /train` — this re-downloads the dataset and re-runs the full pipeline.
5. For programmatic access, send a `POST` request to `/predict` with a JSON body (see API docs below), or explore all endpoints interactively at `http://localhost:8080/docs`.

---

## 9. Screenshots / Demo

*(Add screenshots of the transaction console and result page here once deployed — e.g. a capture of `index.html` and `result.html` in the browser.)*

**Live demo:** _add your EC2 public URL here once deployed, e.g. `http://<ec2-public-ip>:8080`_

---

## 10. API Documentation

### `GET /`
Returns the HTML transaction console.

### `GET /health`
Health check.
**Response:** `{"status": "OK"}`

### `GET /train`
Triggers the full training pipeline (data ingestion → evaluation).
**Response:** `{"message": "Training Successful!"}` or an error payload on failure.

### `POST /predict`
Runs a prediction against a single transaction, returned as JSON.

**Request body** (`application/json`):
```json
{
  "Time": 0.0,
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
  "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
  "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
  "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
  "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
  "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
  "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
  "Amount": 149.62
}
```

**Response** `200 OK`:
```json
{ "response": { "predicted_category": 0 } }
```
`predicted_category` is `0` (legitimate) or `1` (fraud).

**Response** `500` on failure:
```json
{ "error": "<exception message>" }
```

**Authentication:** none currently implemented (see Limitations).

### `POST /predict-form`
Same prediction logic as `/predict`, but accepts `application/x-www-form-urlencoded` data (submitted by `index.html`) and returns a rendered HTML result page instead of JSON.

---

## 11. Engineering Decisions

- **XGBoost over simpler models**: chosen for its strong out-of-the-box performance on tabular data with mixed feature scales (the PCA-transformed `V1`–`V28` features vs. raw `Time`/`Amount`), and native handling of feature interactions without manual feature engineering.
- **SMOTE applied only to the training split**: applying it before the train/test split would leak synthetic points derived from test-set-adjacent data into evaluation, producing misleadingly optimistic metrics. `test.csv` is kept as the untouched, real-world class distribution.
- **Metrics beyond accuracy**: with ~0.172% fraud prevalence, accuracy is close to meaningless — see [Dataset](#2-dataset) and [Model Performance](#12-model-performance). Precision, recall, F1, and ROC-AUC are tracked instead, and a confusion matrix is logged as an artifact so false negatives (missed fraud) are visible at a glance.
- **MLflow via DagsHub instead of a self-hosted tracking server**: avoids managing tracking-server infrastructure. Model *registration* (not just tracking) was deliberately left out of the DagsHub flow after hitting unreliable 500 errors on DagsHub's registry endpoint — models are logged as artifacts instead (`mlflow.xgboost.log_model`), which is sufficient for this project's scale.
- **`mlflow.xgboost` over `mlflow.sklearn` for model logging**: MLflow's sklearn flavor serializes via `skops`, which refuses to trust non-sklearn-native types like `XGBClassifier`/`Booster` by default. The XGBoost-specific flavor uses XGBoost's own serialization and avoids this friction entirely.
- **FastAPI over Flask**: request validation is declarative via Pydantic (catches malformed input before it reaches the model), and Swagger docs are generated automatically — useful both for testing and for documenting the API to other consumers.
- **Config-driven pipeline (YAML-based)**: paths, schema, and hyperparameters live in `config.yaml`/`schema.yaml`/`params.yaml` rather than being hardcoded, so retraining with a different hyperparameter or a schema change doesn't require touching pipeline code.
- **Docker + self-hosted GitHub Actions runner on EC2**: keeps the deployment target simple (a single EC2 instance) while still automating build/push/deploy on every push to `main`, without needing a managed orchestration layer (e.g. ECS/Kubernetes) for a project at this scale.

---

## 12. Model Performance

Evaluated on a held-out test split (untouched by SMOTE, reflecting the real-world ~0.17% fraud rate):

| Metric | Score |
|---|---|
| Accuracy | 99.94% |
| Precision (fraud class) | 0.790 |
| Recall (fraud class) | 0.847 |
| F1-score (fraud class) | 0.818 |
| ROC-AUC | 0.983 |

**Confusion matrix:**

| | Predicted: Legitimate | Predicted: Fraud |
|---|---|---|
| **Actual: Legitimate** | 56,842 | 22 |
| **Actual: Fraud** | 15 | 83 |

**Classification report:**

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.79      0.85      0.82        98

    accuracy                           1.00     56962
   macro avg       0.90      0.92      0.91     56962
weighted avg       1.00      1.00      1.00     56962
```

**Reading these numbers correctly:** accuracy (99.94%) is reported for completeness but, as noted in [Dataset](#2-dataset), is not a meaningful measure on this imbalanced problem — a trivial "always predict legitimate" model would already score 99.83%. The numbers that matter are the fraud-class (`1`) precision/recall: of 98 actual fraud cases in the test set, the model correctly caught **83 (recall 0.847)**, missed **15**, and raised **22 false alarms** out of 56,864 legitimate transactions. The **0.983 ROC-AUC** indicates strong separability between the two classes across decision thresholds, which is where the dataset's authors recommend focusing evaluation (via AUPRC/ROC-AUC) rather than raw accuracy.

---

## 13. Testing

Current testing coverage is intentionally lightweight, focused on catching pipeline-breaking issues early:

- **Schema validation** (`data_validation.py`) acts as a data contract test — every ingested dataset is checked column-by-column against `schema.yaml` before the pipeline proceeds to transformation/training. Result is written to `artifacts/data_validation/status.txt`, and a mismatch halts the pipeline rather than silently training on malformed data.
- **Exploratory notebooks in `research/`** (`01_data_ingestion.ipynb` through `05_model_evaluation.ipynb`) mirror each pipeline stage and were used to validate logic interactively before it was moved into the `src/cred_card_proj` package.
- **CI smoke steps** in `.github/workflows/main.yaml` run placeholder lint/test steps on every push (`Lint code`, `Run unit tests`) — these are stubs intended to be replaced with real checks (e.g. `pytest`, `flake8`) as the test suite grows.

**To run checks locally once a real test suite is added:**
```bash
pytest tests/
```

---

## 14. Limitations & Future Improvements

- **No authentication on the API** — `/predict` and `/train` are open endpoints. Adding API-key or OAuth2 auth is a priority before any public deployment.
- **`/train` runs synchronously and blocks the request** — a long-running retrain should be moved to a background task or a separate job queue rather than blocking the web server.
- **No model versioning/registry** — the current setup overwrites `model.joblib` on every retrain. A registry (or at minimum, versioned artifact paths) would allow rollback to a previous model.
- **No drift or performance monitoring in production** — once deployed, there's no mechanism to detect when the model's real-world performance degrades over time (e.g. via periodic evaluation against fresh labeled data).
- **Single-instance deployment** — the EC2 setup has no load balancing, auto-scaling, or health-check-based restarts. Suitable for a demo/portfolio project, not production traffic.
- **Minimal automated test coverage** — CI currently has placeholder lint/test steps; real unit tests for each pipeline component (ingestion, validation, transformation, evaluation) and API integration tests are a clear next step.
- **No batch prediction endpoint** — only single-transaction predictions are supported; a `/predict-batch` endpoint accepting a CSV or list of transactions would be a natural extension.
- **15 missed fraud cases (false negatives) in evaluation** — for a production fraud system, recall is often prioritized further via threshold tuning or cost-sensitive learning using the `Amount` feature, rather than optimizing for a balanced precision/recall trade-off by default.