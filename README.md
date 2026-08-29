# Cats vs Dogs MLOps Pipeline
```text
Name - ANKITA GOPAKUMAR MEENAKSHI

BITS ID- 2024AC05600
```
End-to-end MLOps implementation for binary image classification of cats and dogs for a pet adoption platform.

This project covers the complete lifecycle from data preparation and model training through experiment tracking, API packaging, containerization, CI/CD deployment, monitoring, logging, and post-deployment performance evaluation.

## Screen Recording Link

https://drive.google.com/file/d/1JGbQQcoHJjcEmIuRkfK-0Chd7Bry6vdy/view?usp=drive_link

## 1. Project Overview

The objective of this project is to implement a reproducible MLOps pipeline for a Cats vs Dogs image classification model.

The pipeline includes:

- Git-based source code versioning
- DVC-based dataset versioning
- Image preprocessing to 224 × 224 RGB
- Train/validation/test split
- Data augmentation
- PyTorch CNN model development
- MLflow experiment tracking
- FastAPI inference service
- Docker containerization
- Unit testing with pytest
- GitHub Actions CI
- Docker Hub image publishing
- GitHub Actions CD
- Docker Compose deployment
- Post-deployment smoke testing
- Prometheus metrics
- Grafana monitoring
- Application logging
- Post-deployment model performance evaluation

## 2. Architecture

```text
                         GitHub
                           |
                      git push main
                           |
                           v
                  GitHub Actions CI
                  +---------------+
                  | pytest        |
                  | Docker build  |
                  | Docker push   |
                  +-------+-------+
                          |
                          v
                      Docker Hub
                          |
                          v
                  GitHub Actions CD
                          |
                          v
                   Docker Compose
          +---------------+---------------+
          |               |               |
          v               v               v
       FastAPI        Prometheus       Grafana
       :8000            :9090           :3000
          |
          v
      CatDogCNN
      model.pt
          |
          v
   Prediction + Logs

Data / Model Development Flow

Kaggle Cats & Dogs Dataset
          |
          v
   Image Preprocessing
   224x224 RGB
          |
          v
     80 / 10 / 10
 Train / Val / Test
          |
          v
       PyTorch CNN
          |
          v
        MLflow
 Params / Metrics / Artifacts
          |
          v
      models/model.pt
```

## 3. Technology Stack

| Area | Tool |
|---|---|
| Source Control | Git + GitHub |
| Dataset Versioning | DVC |
| Model Development | PyTorch |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Testing | pytest |
| Containerization | Docker |
| Container Registry | Docker Hub |
| Continuous Integration | GitHub Actions |
| Continuous Deployment | GitHub Actions |
| Deployment Target | Docker Compose |
| Metrics | Prometheus |
| Dashboard | Grafana |
| Logging | Python logging |

## 4. Project Structure

```text
cats-dogs-mlops/
|
|-- .github/
|   `-- workflows/
|       |-- ci.yml
|       `-- cd.yml
|
|-- artifacts/
|   |-- confusion_matrix.png
|   |-- training_accuracy_curve.png
|   |-- training_loss_curve.png
|   |-- training_history.json
|   |-- test_metrics.json
|   `-- deployed_performance.json
|
|-- data/
|   |-- raw/
|   |-- raw.dvc
|   |-- processed/
|   `-- processed.dvc
|
|-- deployment/
|   `-- docker-compose.yml
|
|-- models/
|   `-- model.pt
|
|-- monitoring/
|   `-- prometheus.yml
|
|-- sample_images/
|   |-- cat.jpg
|   `-- dog.jpg
|
|-- scripts/
|   |-- smoke_test.py
|   `-- evaluate_deployed.py
|
|-- src/
|   |-- __init__.py
|   |-- app.py
|   |-- inference.py
|   |-- model.py
|   |-- preprocess.py
|   `-- train.py
|
|-- tests/
|   |-- test_inference.py
|   `-- test_preprocess.py
|
|-- .dockerignore
|-- .gitignore
|-- Dockerfile
|-- pytest.ini
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

## 5. Dataset

The project uses the Cats and Dogs binary image classification dataset from Kaggle.

Images are organized into two classes:

```text
cats
dogs
```

### Preprocessing

Images are:

- Converted to RGB
- Resized to 224 × 224 pixels
- Split into:
  - 80% training
  - 10% validation
  - 10% testing

Training data augmentation includes:

- Random horizontal flip
- Random rotation
- Color jitter

The processed data is versioned using DVC.

## 6. Prerequisites and Initial Repository Setup

Install the following before starting:

- Python 3.11
- Git
- Docker Desktop
- A GitHub account
- A Docker Hub account

Verify:

```powershell
python --version
git --version
docker --version
docker compose version
```

Create the project folder and initialize Git:

```powershell
mkdir cats-dogs-mlops
cd cats-dogs-mlops
git init
git branch -M main
```

Create the main project folders:

```powershell
mkdir src
mkdir data
mkdir data\raw
mkdir data\processed
mkdir models
mkdir artifacts
mkdir tests
mkdir deployment
mkdir monitoring
mkdir scripts
mkdir sample_images
mkdir .github
mkdir .github\workflows
```

Create an empty Python package file:

```powershell
New-Item src\__init__.py -ItemType File
```

Create a GitHub repository named:

```text
cats-dogs-mlops
```

Then connect the local repository:

```powershell
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/cats-dogs-mlops.git
git add .
git commit -m "Initialize Cats vs Dogs MLOps project"
git push -u origin main
```

## 7. Environment Setup

### Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

## 8. Dataset Download and Data Versioning with DVC

Download the Cats and Dogs binary classification dataset from Kaggle.

After extraction, organize the raw files as:

```text
data/raw/
|-- cats/
|   |-- cat.0.jpg
|   `-- ...
`-- dogs/
    |-- dog.0.jpg
    `-- ...
```

The preprocessing code assumes the class folders are named exactly:

```text
cats
dogs
```

Install and initialize DVC:

```powershell
pip install dvc
dvc init
```

Track the raw dataset:

```powershell
dvc init
```

Track raw data:

```powershell
dvc add data/raw
```

Track processed data:

```powershell
dvc add data/processed
```

The `.dvc` pointer files are committed to Git while the large dataset folders remain outside normal Git tracking.

The trained `models/model.pt` is kept directly in Git so the GitHub Actions runner can build the Docker image without requiring a separately configured DVC remote. DVC is used for the required dataset versioning.

Check DVC status:

```powershell
dvc status
```

## 9. Data Preprocessing

Run preprocessing from the project root:

```powershell
python src/preprocess.py
```

Expected output includes train, validation, and test image counts for both classes.

Example class structure:

```text
data/processed/
|-- train/
|   |-- cats/
|   `-- dogs/
|-- val/
|   |-- cats/
|   `-- dogs/
`-- test/
    |-- cats/
    `-- dogs/
```

## 10. Model Development

The baseline model is a custom convolutional neural network implemented in PyTorch.

### CNN Architecture

```text
Input 224x224 RGB
       |
       v
Conv2D 32 + ReLU + MaxPool
       |
       v
Conv2D 64 + ReLU + MaxPool
       |
       v
Conv2D 128 + ReLU + MaxPool
       |
       v
Adaptive Average Pool
       |
       v
Dense 128 + ReLU + Dropout
       |
       v
Binary Output
```

The model uses:

- `BCEWithLogitsLoss`
- Adam optimizer
- Sigmoid during inference
- Binary class mapping:
  - `cats = 0`
  - `dogs = 1`

### Train the model

```powershell
python -m src.train
```

The best model is saved to:

```text
models/model.pt
```

## 11. Experiment Tracking with MLflow

MLflow tracks:

### Parameters

- Model architecture
- Image size
- Batch size
- Number of epochs
- Learning rate
- Optimizer
- Loss function
- Random seed
- Dataset sizes

### Metrics

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy
- Test loss
- Test accuracy
- Precision
- Recall
- F1 score

### Artifacts

- `model.pt`
- Training loss curve
- Training accuracy curve
- Confusion matrix
- Test metrics JSON
- Training history JSON

The project uses a local SQLite MLflow backend. Newer MLflow versions may reject the old filesystem tracking backend, so the training code should use:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

Start MLflow on Windows using a single worker:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Open:

```text
http://127.0.0.1:5000
```

## 12. FastAPI Inference Service

The trained model is exposed through FastAPI.

Start the API locally:

```powershell
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Health Endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "CatDogCNN"
}
```

### Prediction Endpoint

```http
POST /predict
```

The endpoint accepts a JPEG or PNG image and returns:

- Predicted class
- Confidence
- Cat probability
- Dog probability
- Filename

Example:

```json
{
  "label": "cat",
  "confidence": 0.89,
  "cat_probability": 0.89,
  "dog_probability": 0.11,
  "filename": "cat.jpg"
}
```

### Metrics Endpoint

```http
GET /metrics
```

Prometheus metrics include:

- `api_requests_total`
- `prediction_requests_total`
- `prediction_errors_total`
- `prediction_latency_seconds`

## 13. Test the API

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Prediction:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/predict" `
  -F "file=@sample_images/cat.jpg"
```

## 14. Docker

Before building, verify that `models/model.pt` exists:

```powershell
Test-Path models\model.pt
```

It should return:

```text
True
```

`requirements.txt` should contain pinned inference/runtime dependencies, for example:

```text
torch==<installed-version>
torchvision==<installed-version>
fastapi==<installed-version>
uvicorn==<installed-version>
python-multipart==<installed-version>
Pillow==<installed-version>
prometheus-client==<installed-version>
```

Use the exact versions installed in the development environment:

```powershell
pip show torch torchvision fastapi uvicorn python-multipart pillow prometheus-client
```

The Dockerfile should use a single-line JSON `CMD`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models/model.pt ./models/model.pt

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build

```powershell
docker build -t cats-dogs-api:1.0 .
```

### Run

```powershell
docker run `
  --name cats-dogs-api `
  -p 8000:8000 `
  cats-dogs-api:1.0
```

### Verify

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 15. Automated Testing

The project includes tests for:

- Image preprocessing
- Inference preprocessing
- Prediction utility behavior

Create `pytest.ini` in the repository root so the `src` package is importable during local and CI tests:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

Run:

```powershell
pytest -v
```

Expected:

```text
tests/test_inference.py::test_inference_preprocessing PASSED
tests/test_inference.py::test_predict_image PASSED
tests/test_preprocess.py::test_preprocess_image PASSED
```

## 16. Continuous Integration

GitHub Actions CI runs on:

- Push to `main`
- Pull requests to `main`

The CI pipeline:

```text
Checkout
   |
   v
Set up Python
   |
   v
Install Dependencies
   |
   v
pytest
   |
   v
Docker Build
   |
   v
Docker Hub Login
   |
   v
Docker Image Push
```

Images are tagged using:

```text
latest
<Git commit SHA>
```

This provides traceability between source commits and container images.

## 17. Docker Hub and GitHub CI Credentials

Create a Docker Hub repository named:

```text
cats-dogs-api
```

Use Public visibility for the simplest assignment setup.

Create a Docker Hub Personal Access Token with Read/Write access:

```text
Docker Hub
-> Account Settings
-> Personal access tokens
-> Generate new token
```

Do not place the token in source code or YAML.

The CI workflow publishes the API image to:

```text
DOCKERHUB_USERNAME/cats-dogs-api:latest
```

Docker Hub credentials are stored using GitHub repository secrets and variables.

Required GitHub repository configuration:

```text
Repository
-> Settings
-> Secrets and variables
-> Actions
```

Create:

```text
Secret:
DOCKERHUB_TOKEN

Variable:
DOCKERHUB_USERNAME
```

The Docker Hub image is tagged as both:

```text
DOCKERHUB_USERNAME/cats-dogs-api:latest
DOCKERHUB_USERNAME/cats-dogs-api:<git-commit-sha>
```

## 18. Continuous Deployment

The CD pipeline uses a Windows self-hosted GitHub Actions runner.

The deployment flow is:

```text
CI Success
   |
   v
CD Starts
   |
   v
Checkout Commit
   |
   v
Docker Compose Pull
   |
   v
Docker Compose Up
   |
   v
Wait for Startup
   |
   v
Health Smoke Test
   |
   v
Prediction Smoke Test
```

The CD workflow deploys only after the CI pipeline completes successfully.

### Configure the Windows self-hosted runner

In GitHub:

```text
Repository
-> Settings
-> Actions
-> Runners
-> New self-hosted runner
-> Windows
-> x64
```

Install the runner under:

```text
C:\actions-runner
```

Do not install it under `C:\Windows\System32`.

Follow the repository-specific download and registration commands shown by GitHub.

For this assignment, the simplest reliable approach is to configure the runner without installing it as a Windows service, then start it manually:

```powershell
cd C:\actions-runner
.\run.cmd
```

Keep this window open while running CD. GitHub should show the runner as:

```text
Idle
```

Docker Desktop must also be running.

Verify:

```powershell
docker info
docker compose version
```

The CD workflow should use the runner labels:

```yaml
runs-on:
  - self-hosted
  - Windows
  - X64
```

The CD workflow is triggered after the `MLOps CI Pipeline` completes successfully on `main`.

## 19. Docker Compose Deployment

The final `deployment/docker-compose.yml` should deploy the API, Prometheus, and Grafana on one explicitly named Docker network.

Example structure:

```yaml
services:
  cats-dogs-api:
    image: YOUR_DOCKERHUB_USERNAME/cats-dogs-api:latest
    ports:
      - "8000:8000"
    networks:
      - mlops-network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v3.14.0
    ports:
      - "9090:9090"
    volumes:
      - ../monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    networks:
      - mlops-network
    depends_on:
      - cats-dogs-api
    restart: unless-stopped

  grafana:
    image: grafana/grafana:13.2.0
    ports:
      - "3000:3000"
    networks:
      - mlops-network
    depends_on:
      - prometheus
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

networks:
  mlops-network:
    name: cats-dogs-mlops-network
    driver: bridge

volumes:
  grafana-data:
```

Replace `YOUR_DOCKERHUB_USERNAME` with the actual Docker Hub username.

Start the full stack manually:

```powershell
docker compose -f deployment\docker-compose.yml up -d
```

Force recreation:

```powershell
docker compose -f deployment\docker-compose.yml up -d --force-recreate --remove-orphans
```

If Docker reports that the container name is already in use, locate and remove the previous standalone container before starting Compose:

```powershell
docker ps -a
docker stop cats-dogs-api
docker rm cats-dogs-api
```

Stop:

```powershell
docker compose -f deployment\docker-compose.yml down
```

Check containers:

```powershell
docker ps
```

Expected services:

```text
cats-dogs-api
prometheus
grafana
```

## 20. Smoke Tests

The CD smoke test expects a committed test image at:

```text
sample_images/cat.jpg
```

Verify it is tracked by Git:

```powershell
git status
git add sample_images/cat.jpg
git commit -m "Add smoke test image"
git push origin main
```

The smoke test uses:

```python
BASE_URL = "http://127.0.0.1:8000"
```

Run:

```powershell
python scripts\smoke_test.py
```

The smoke test validates:

1. `/health` returns HTTP 200 and reports the service as healthy.
2. `/predict` accepts an image and returns the required prediction fields.

The script exits with a non-zero status if either check fails, causing the CD pipeline to fail.

## 21. Monitoring with Prometheus

Prometheus is available at:

```text
http://127.0.0.1:9090
```

The API is scraped from the Docker Compose service name:

```text
http://cats-dogs-api:8000/metrics
```

`monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "cats-dogs-api"
    metrics_path: /metrics
    static_configs:
      - targets:
          - "cats-dogs-api:8000"
```

Validate the target at:

```text
http://127.0.0.1:9090/targets
```

The `cats-dogs-api` target should be `UP`.

Useful PromQL queries:

### Service health

```promql
up{job="cats-dogs-api"}
```

### Total prediction requests

```promql
prediction_requests_total
```

### Requests over the last five minutes

```promql
increase(prediction_requests_total[5m])
```

### Request rate

```promql
rate(prediction_requests_total[5m])
```

### Average prediction latency

```promql
rate(prediction_latency_seconds_sum[5m])
/
rate(prediction_latency_seconds_count[5m])
```

### Prediction errors

```promql
prediction_errors_total
```

## 22. Grafana

Grafana is available at:

```text
http://127.0.0.1:3000
```

Default first login is typically:

```text
Username: admin
Password: admin
```

Add Prometheus as the Grafana data source:

```text
Connections
-> Data sources
-> Add data source
-> Prometheus
```

Prometheus server URL:

```text
http://prometheus:9090
```

Click `Save & test`.

Create a dashboard:

```text
Dashboards
-> New
-> New dashboard
-> Add visualization
```

Recommended dashboard panels:

| Panel | PromQL |
|---|---|
| Total Prediction Requests | `prediction_requests_total` |
| Requests in Last 5 Minutes | `increase(prediction_requests_total[5m])` |
| Prediction Request Rate | `rate(prediction_requests_total[5m])` |
| Average Prediction Latency | `rate(prediction_latency_seconds_sum[5m]) / rate(prediction_latency_seconds_count[5m])` |
| Prediction Errors | `prediction_errors_total` |

Suggested dashboard name:

```text
Cats vs Dogs MLOps Monitoring
```

## 23. Application Logging

The FastAPI service logs:

- Model startup
- Prediction completion
- Predicted label
- Confidence
- Request latency
- Invalid file types
- Prediction errors

Example:

```text
INFO | Prediction completed label=cat confidence=0.8912 latency=0.0821s
```

Uploaded image contents are not written to application logs.

View Docker logs:

```powershell
docker compose `
  -f deployment\docker-compose.yml `
  logs --tail=30 cats-dogs-api
```

## 24. Post-Deployment Performance Evaluation

The deployed API is evaluated using labeled images from the test dataset.

The evaluation script sends:

- 20 cat images
- 20 dog images

through the deployed `/predict` endpoint.

Run:

```powershell
python scripts\evaluate_deployed.py
```

The script calculates:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Results are stored in:

```text
artifacts/deployed_performance.json
```

This verifies performance of the deployed service rather than only the locally loaded model.

## 25. Generate Monitoring Traffic

Example:

```powershell
1..20 | ForEach-Object {
    curl.exe -s -X POST `
      "http://127.0.0.1:8000/predict" `
      -F "file=@sample_images/cat.jpg"
}
```

Then verify:

```powershell
curl.exe -s http://127.0.0.1:8000/metrics |
Select-String "prediction_requests_total"
```

If FastAPI shows a non-zero counter but Prometheus shows a different value, compare the endpoint from the Prometheus container:

```powershell
docker ps --filter "ancestor=prom/prometheus" --format "{{.Names}}"
```

Then:

```powershell
docker exec <PROMETHEUS_CONTAINER_NAME> wget -qO- http://cats-dogs-api:8000/metrics |
Select-String "prediction_requests_total"
```

The host and Prometheus-container values should match. If they do not, remove old/orphaned API containers and recreate the stack on the explicit `cats-dogs-mlops-network`.

## 26. Key URLs

| Service | URL |
|---|---|
| FastAPI | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| Health | `http://127.0.0.1:8000/health` |
| Metrics | `http://127.0.0.1:8000/metrics` |
| MLflow | `http://127.0.0.1:5000` |
| Prometheus | `http://127.0.0.1:9090` |
| Grafana | `http://127.0.0.1:3000` |

## 27. MLOps Pipeline Summary

### M1 - Model Development and Experiment Tracking

Implemented:

- Git source control
- DVC data versioning
- Image preprocessing
- 80/10/10 dataset split
- Data augmentation
- PyTorch CNN
- Model serialization
- MLflow parameters, metrics, and artifacts

### M2 - Model Packaging and Containerization

Implemented:

- FastAPI inference service
- `/health`
- `/predict`
- Version-pinned requirements
- Dockerfile
- Local Docker validation

### M3 - Continuous Integration

Implemented:

- Preprocessing unit test
- Inference unit tests
- pytest
- GitHub Actions CI
- Docker image build
- Docker Hub publishing
- Commit-SHA image versioning

### M4 - Continuous Deployment

Implemented:

- Docker Compose deployment
- GitHub Actions CD
- Windows self-hosted runner
- Automatic image pull/update
- Health smoke test
- Prediction smoke test
- Pipeline failure on smoke-test failure

### M5 - Monitoring and Post-Deployment Performance

Implemented:

- Request logging
- Prediction logging
- Request counter
- Error counter
- Latency histogram
- Prometheus
- Grafana
- Post-deployment labeled evaluation
- Performance JSON artifact


## 28. Reproduction Checklist

For a clean machine or evaluator, the intended reproduction order is:

1. Clone the GitHub repository.
2. Create and activate the Python 3.11 virtual environment.
3. Install `requirements-dev.txt`.
4. Place/download the Kaggle dataset under `data/raw/cats` and `data/raw/dogs`, or restore it through the configured DVC storage if a DVC remote is available.
5. Run preprocessing if processed data is not restored.
6. Train the CNN using `python -m src.train`, or use the included `models/model.pt`.
7. Start MLflow with the SQLite backend if experiment history is required.
8. Run `pytest -v`.
9. Build and test the Docker image.
10. Configure Docker Hub and GitHub Actions secrets/variables.
11. Start the Windows self-hosted runner and Docker Desktop.
12. Push to `main` to execute CI followed by CD.
13. Verify the Docker Compose API deployment.
14. Verify Prometheus target health.
15. Configure the Grafana Prometheus datasource and dashboard.
16. Run `scripts/evaluate_deployed.py` for post-deployment model performance.

## 29. Deliverables

The final submission contains:

- Source code
- DVC configuration
- CI/CD workflows
- Dockerfile
- Docker Compose manifest
- Prometheus configuration
- Trained model artifact
- Test scripts
- Smoke-test script
- Post-deployment evaluation script
- Experiment artifacts
- Monitoring artifacts
- README

A separate screen recording demonstrates the complete MLOps workflow from a code change through automated deployment and prediction.
