# Screenshot register

Every screenshot is a real capture of a real run. Save each file here with **exactly** the name below.
Where it says *Prayogshala*, the browser address bar / lab banner must be visible in the capture.
Tick ✅ once the file exists and has been checked.

| # | File name | Criterion | What it must show | Where captured | Report fig. | ✅ |
|---|---|---|---|---|---|---|
| 01 | `01_prayogshala_env.png` | 4 | Prayogshala lab session: terminal with `hostname`, `python --version`, `pip list` head | Prayogshala | 1 | ✅ |
| 02 | `02_data_download.png` | 1 | Output of `python scripts/download_data.py` (rows, SHA-256) | Prayogshala | 2 | ✅ |
| 03 | `03_data_cleaning.png` | 1 | Output of `python -m heartrisk.data` (missing values, duplicates, class counts) | Prayogshala | 2 | ✅ |
| 04 | `04_eda_notebook.png` | 1 | `01_eda.ipynb` executed: histograms visible | Prayogshala | 3 | ✅ |
| 05 | `05_eda_heatmap_balance.png` | 1 | Correlation heatmap + class balance in the notebook | Prayogshala | 3 | ✅ |
| 06 | `06_training_run.png` | 2 | Terminal/notebook output of training (CV ROC-AUC per model, selected model) | Prayogshala | 4 | ✅ |
| 07 | `07_mlflow_experiment_runs.png` | 3 | MLflow UI: experiment `heart-disease-uci` with the 3 parent runs + selection run | MLflow UI | 5 | ✅ |
| 08 | `08_mlflow_run_detail.png` | 3 | MLflow run page: parameters + metrics of the selected model | MLflow UI | 5 | ✅ |
| 09 | `09_mlflow_artifacts_plots.png` | 3 | MLflow run artifacts: ROC curve / confusion matrix plot + logged model | MLflow UI | 5 | ✅ |
| 10 | `10_mlflow_compare.png` | 3 | MLflow "Compare" view of the 3 model runs (metrics side by side) | MLflow UI | 5 | ✅ |
| 10b | `10b_mlflow_compare_metrics.png` | 3 | MLflow "Compare": CV metric rows (accuracy, precision, recall, F1 ± std) for the 3 models | MLflow UI | 5 | ✅ |
| 11 | `11_mlflow_registry.png` | 4 | Model registry: `heart-risk-classifier` with alias `champion` | MLflow UI | 6 | ✅ |
| 12 | `12_pytest_local.png` | 5 | `pytest --cov=heartrisk` all passed + coverage table | Prayogshala | 7 | ✅ |
| 13 | `13_flake8_clean.png` | 5 | `flake8 src test scripts` with no output, `echo $?` → 0 | Prayogshala | 7 | ✅ |
| 14 | `14_github_repo.png` | 9 | GitHub repository home page (file tree, commits) | GitHub | 8 | ✅ |
| 15 | `15_actions_success.png` | 5 | GitHub Actions run #3: all 4 jobs green | GitHub | 7 | ✅ |
| 16 | `16_actions_artifacts.png` | 5 | Run #4 summary: metrics table + 5 uploaded artifacts (pinned ubuntu-24.04, 0 warnings) | GitHub | 7 | ✅ |
| 17 | `17_actions_failure.png` | 5 | PR #1 (deliberately failing test, branch `demo/ci-failure`): lint ✓, tests ✗, training + Docker skipped | GitHub | 7 | ✅ |
| 17b | `17b_actions_failure_log.png` | 5 | Failing-test trace in the CI log (`AssertionError … assert 3 == 2`) | GitHub | 7 | ✅ |
| 17c | `17c_actions_failure_summary.png` | 5 | `1 failed, 53 passed`, `Process completed with exit code 1`; test results still uploaded | GitHub | 7 | ✅ |
| 18 | `18_docker_build.png` | 6 | `docker build --no-cache`: 14/14 FINISHED, 7 Dockerfile steps (rebuilt after disk incident) | Laptop | 9 | ✅ |
| 18b | `18b_docker_nonroot_user.png` | 6 | `docker images` (image 4746b2a256ba, 163 MB) + `id` → uid=10001(appuser) non-root | Laptop | 9 | ✅ |
| 19 | `19_docker_run_predict.png` | 6 | Container Up (healthy); `/health`; `/predict` high-risk 1/0.9615, low-risk 0/0.038 | Laptop | 9 | ✅ |
| 20 | `20_docker_invalid_input.png` | 6 | HTTP 422 with per-field errors + structured JSON container logs (no patient data) | Laptop | 9 | ✅ |
| 21 | `21_swagger_docs.png` | 6 | `/docs` OpenAPI page: 5 endpoints + schemas | Laptop | 9 | ✅ |
| 22 | `22_k8s_apply.png` | 7 | `kubectl apply -k deploy/`: 15 resources created; API + Grafana rolled out, Prometheus first timed out (image pull) | Laptop | 10 | ✅ |
| 22b | `22b_k8s_rollout_ok.png` | 7 | All three deployments `successfully rolled out` | Laptop | 10 | ✅ |
| 22c | `22c_prometheus_pull_recovery.png` | 7 | Pod events: transient registry/network pull errors → back-off → `Successfully pulled` (self-healing) | Laptop | 10 | ✅ |
| 23 | `23_k8s_pods_services.png` | 7 | `get deploy,pods,svc -o wide`: API 2/2, 4 pods Running 0 restarts, 3 LoadBalancer services on localhost (8080/9090/3000) | Laptop | 10 | ✅ |
| 24 | `24_k8s_predict.png` | 7 | `/health` + `/predict` through the LoadBalancer :8080 → 1/0.9615 and 0/0.038 (identical to Docker) | Laptop | 10 | ✅ |
| 25 | `25_k8s_describe.png` | 7 | `describe deployment`: 2 replicas, RollingUpdate 0 unavailable/1 surge, limits/requests, liveness/readiness/startup probes, Prometheus annotations | Laptop | 10 | ✅ |
| 26 | `26_api_logs.png` | 8 | K8s pod logs filtered to /predict: structured JSON request lines (200/422, latency) + prediction events (class, confidence, model version), no patient data | Laptop | 11 | ✅ |
| 26b | `26b_api_logs_both_pods.png` | 8 | Logs from both replicas (`--prefix`): request id, path, status, latency for probe traffic | Laptop | 11 | ✅ |
| 27 | `27_prometheus_targets.png` | 8 | Prometheus targets: heart-risk-api 2/2 UP (Kubernetes pod discovery, pod/env labels) | Laptop | 11 | ✅ |
| 28 | `28_grafana_dashboard.png` | 8 | Grafana: 547 predictions, 0.55 req/s, 7.3 % error ratio, 2 pods up, rate by endpoint/status, latency p50/p95, class mix, errors by reason | Laptop | 11 | ✅ |
| 28b | `28b_grafana_confidence.png` | 8 | Grafana: mean prediction confidence over time (≈96 % samples → ≈78 % synthetic traffic) | Laptop | 11 | ✅ |
