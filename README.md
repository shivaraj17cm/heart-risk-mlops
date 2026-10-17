# Heart Disease Risk — End-to-End MLOps Pipeline

| | |
|---|---|
| **Course** | MLOps — AIMLZG523 (S1-26), BITS Pilani WILP |
| **Assignment** | Assignment 1 — End-to-End ML Model Development, CI/CD and Production Deployment |
| **Student** | SHIVARAJA C M |
| **Student number** | 2025AE05109 |
| **Email** | 2025AE05109@wilp.bits-pilani.ac.in |
| **Contribution** | 100 % (individual assignment) |

A reproducible pipeline that predicts the presence of heart disease from the **UCI Heart Disease
(Cleveland)** dataset. It covers data acquisition and EDA, feature engineering, three tuned models,
MLflow experiment tracking, a packaged model, automated tests with GitHub Actions CI/CD, a Dockerised
FastAPI service, Kubernetes deployment, and Prometheus + Grafana monitoring.

> ⚠️ Academic exercise only. The model is **not** a clinically validated diagnostic tool and must not be used for medical decisions.

## Results at a glance (real UCI Cleveland data, run in the BITS Prayogshala lab)

| Model | CV ROC-AUC (5-fold) | Test ROC-AUC | Test accuracy | Test precision | Test recall |
|---|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.907 ± 0.019** | **0.959** | **0.885** | **0.818** | **0.964** |
| Random Forest | 0.901 ± 0.025 | 0.960 | 0.902 | 0.867 | 0.929 |
| Gradient Boosting | 0.876 ± 0.027 | 0.953 | 0.902 | 0.867 | 0.929 |

Selection rule: highest mean CV ROC-AUC (test set never used for selection); LR also has the smallest
train-vs-CV gap (0.024). Identical numbers were reproduced by GitHub Actions on Python 3.11.
Full write-up: [`report/MLOps_Assignment1_Report_2025AE05109.pdf`](report/MLOps_Assignment1_Report_2025AE05109.pdf).

---

## 1. Architecture

```
 UCI repository ──► scripts/download_data.py ──► data/raw/  (+ manifest.json, SHA-256)
                               │
             heartrisk.data: schema checks · '?'→NaN · dedupe · range checks · binary target
                               │
                     data/processed/heart_clean.csv ──► heartrisk.eda ──► reports/figures/eda/
                               │
  heartrisk.train ── sklearn Pipeline [ClinicalFeatureBuilder → ColumnTransformer → classifier]
     • LogisticRegression · RandomForest · GradientBoosting
     • GridSearchCV + StratifiedKFold(5) on the training split, stratified 20 % hold-out test
     • MLflow: one parent run per model, one child run per hyper-parameter set,
       selection run, model registry alias "champion"
                               │
             artifacts/model/model.joblib + model_metadata.json
                               │
  FastAPI (/predict /health /ready /metrics) ──► Docker image ──► Kubernetes (2 replicas,
  LoadBalancer Service, probes, limits) ──► Prometheus (pod discovery) ──► Grafana dashboard

  GitHub Actions: lint ─► pytest ─► data+EDA+train ─► docker build + smoke test  (artifacts per run)
```

## 2. Repository layout

| Path | Contents |
|---|---|
| `scripts/download_data.py` | Reproducible dataset download (direct file → official ZIP fallback, retries, checksum manifest) |
| `src/heartrisk/` | Python package: `data`, `features`, `train`, `evaluate`, `eda`, `model_io`, `predict`, `api/` |
| `configs/train.yaml` | Seeds, CV folds, hyper-parameter grids, selection rule (all logged to MLflow) |
| `notebooks/` | `01_eda`, `02_training_mlflow`, `03_inference` (they call the package, so results match the scripts) |
| `test/` | Pytest suite (data, features, model + MLflow, API, scripts) — offline, synthetic UCI-format fixture |
| `data/raw/`, `data/processed/` | Raw UCI file + manifest, cleaned CSV |
| `artifacts/model/` | Packaged pipeline (`model.joblib`) and metadata |
| `reports/` | EDA summary, metrics, model comparison, figures |
| `Dockerfile`, `requirements-serve.txt` | Serving image (slim, non-root, health check) |
| `docker-compose.yml` | Local API + Prometheus + Grafana stack |
| `deploy/` | Kubernetes manifests (`k8s/`), monitoring configs, `kustomization.yaml` |
| `.github/workflows/ci.yml` | CI/CD pipeline |
| `screenshots/` | Evidence screenshots (see `screenshots/README.md`) |
| `report/` | Final report (PDF) |

## 3. Prerequisites

* Python **3.10–3.12** (CI and Docker use 3.11), Git
* Docker Desktop (for the container and Kubernetes parts) with Kubernetes enabled, or Minikube
* ~4 GB free RAM for the full Kubernetes + monitoring stack

## 4. Setup (clean environment)

```bash
git clone https://github.com/shivaraj17cm/heart-risk-mlops.git
cd heart-risk-mlops
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt      # pinned versions; also installs this package (-e .)
```

## 5. Run the whole pipeline

```bash
bash scripts/run_pipeline.sh                                        # Linux / macOS / Prayogshala
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1  # Windows
```

Or step by step:

```bash
python scripts/download_data.py          # data/raw/processed.cleveland.data + manifest.json
python -m heartrisk.data                 # data/processed/heart_clean.csv + reports/cleaning_report.json
python -m heartrisk.eda                  # reports/figures/eda/*.png + reports/eda_summary.json
python -m heartrisk.train                # MLflow runs + artifacts/model/ + reports/metrics.json
python -m heartrisk.predict --input samples/sample_request.json
```

### Dataset
UCI Machine Learning Repository, *Heart Disease* (id 45), Cleveland file `processed.cleveland.data`
— 303 patients, 13 attributes + diagnosis `num` (0–4), `?` = missing (in `ca` and `thal`).
Target: `target = 1 if num > 0 else 0`. Licence CC BY 4.0. The script records the SHA-256 checksum,
so later runs reuse the verified copy and you can see if the upstream file ever changes.

## 6. Experiment tracking (MLflow)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # open http://127.0.0.1:5000
```
Experiment **heart-disease-uci**: a parent run per model family (best parameters, CV mean±std,
hold-out metrics, ROC / confusion-matrix / feature-importance plots, logged model), child runs for every
grid combination, and a `model-selection` run. The selected model is registered as
`heart-risk-classifier` with alias `champion`.

## 7. Tests and linting

```bash
flake8 src test scripts
pytest --cov=heartrisk
```

## 8. Docker

```bash
docker build -t heart-risk-api:1.0.0 .
docker run -d --name heart-risk-api -p 8000:8000 heart-risk-api:1.0.0
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/sample_request.json
```
PowerShell: `Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -InFile samples\sample_request.json`

### API contract
`POST /predict` — body: the 13 UCI attributes (`ca`, `thal` may be `null`; they are imputed).

```json
{"age": 58, "sex": 1, "cp": 4, "trestbps": 132, "chol": 224, "fbs": 0, "restecg": 2,
 "thalach": 173, "exang": 0, "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 7}
```
Response:
```json
{"prediction": 1, "label": "heart disease present", "probability_disease": 0.9615,
 "confidence": 0.9615, "model_version": "1.0.0+b00a9c9", "request_id": "…"}
```
`confidence` is the probability of the predicted class. Invalid input → HTTP 422 with a per-field
error list. Other endpoints: `GET /health` (liveness), `GET /ready` (model loaded), `GET /metrics`
(Prometheus), `GET /docs` (OpenAPI UI).

## 9. Kubernetes deployment (Docker Desktop or Minikube)

```bash
docker build -t heart-risk-api:1.0.0 .
# Minikube only:  minikube image load heart-risk-api:1.0.0
kubectl apply -k deploy/
kubectl -n heart-risk rollout status deployment/heart-risk-api
kubectl -n heart-risk get pods,svc
curl http://localhost:8080/health          # Minikube: run `minikube tunnel`, use the EXTERNAL-IP
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @samples/sample_request.json
```
The API Service is `type: LoadBalancer` (port 8080 → container 8000). An optional NGINX Ingress
is in `deploy/k8s/optional/ingress.yaml`. Remove everything with `kubectl delete -k deploy/`.

## 10. Monitoring and logging

* Every request is logged as one JSON line (request id, method, path, status, latency); every
  prediction logs class, confidence and model version. **Clinical feature values are never logged.**
* `/metrics` exposes request count, latency histogram, predictions by class, confidence, errors, model info.
* Prometheus (http://localhost:9090) discovers every API pod; Grafana (http://localhost:3000,
  admin/admin) auto-loads the **Heart Risk API – Service Monitoring** dashboard.
* Generate traffic: `python scripts/send_traffic.py --url http://localhost:8080 --n 300`
* Without Kubernetes: `docker compose up -d --build` starts API + Prometheus + Grafana.

## 11. CI/CD
`.github/workflows/ci.yml` runs on every push / pull request to `main`:
**lint → unit tests (+coverage) → data, EDA and training → Docker build and smoke test.**
Each job uploads its artifacts (lint report, JUnit/coverage, model, figures, MLflow store, logs).
Every step runs in bash with `-eo pipefail`, so any lint, test, training or smoke-test failure turns
the run red and stops the jobs that depend on it.

## 12. Reproducibility controls
Pinned dependency versions · fixed seeds (split, CV, models) · checksum-verified data · all learned
preprocessing inside one serialized pipeline · config and data hash logged to MLflow · relative paths
only · the same tests run locally and in CI.

## 13. API access (deliverable: "deployed API URL or access instructions")

The API is deployed on **local Kubernetes (Docker Desktop)**; no public cloud endpoint was created.
After `docker build -t heart-risk-api:1.0.0 .` and `kubectl apply -k deploy/` it is reachable at:

| Endpoint | URL |
|---|---|
| Prediction | `POST http://localhost:8080/predict` |
| Health / readiness | `GET http://localhost:8080/health`, `GET http://localhost:8080/ready` |
| OpenAPI docs | `http://localhost:8080/docs` |
| Prometheus / Grafana | `http://localhost:9090` / `http://localhost:3000` |

Example (PowerShell: use `curl.exe`):
```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @samples/sample_request.json
# {"prediction":1,"label":"heart disease present","probability_disease":0.9615,"confidence":0.9615,"model_version":"1.0.0+b00a9c9",...}
```

## 14. Evidence, report and execution environments

* `screenshots/` — 36 genuine captures indexed in [`screenshots/README.md`](screenshots/README.md)
  (Prayogshala lab: 01–13 · GitHub/CI: 14–17c · Docker: 18–21 · Kubernetes: 22–25 · Monitoring: 26–28b).
  Prayogshala session tokens visible in the browser address bar were redacted.
* `report/MLOps_Assignment1_Report_2025AE05109.pdf` — 10-page report.
* Executed notebooks in `notebooks/` contain the outputs of the Prayogshala run.
* Environments: Prayogshala VM (data, EDA, training, MLflow, tests) · GitHub Actions (CI/CD) ·
  Windows laptop with Docker Desktop 29.3 and Kubernetes v1.34.1 (container, deployment, monitoring).
