# Cats vs Dogs MLOps Pipeline

End-to-end MLOps implementation for binary image classification of cats and dogs for a pet adoption platform.

This project covers the complete lifecycle from data preparation and model training through experiment tracking, API packaging, containerization, CI/CD deployment, monitoring, logging, and post-deployment performance evaluation.

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

## 6. Environment Setup

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

## 7. Data Versioning with DVC

Initialize DVC:

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

Check DVC status:

```powershell
dvc status
```

## 8. Data Preprocessing

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

## 9. Model Development

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

## 10. Experiment Tracking with MLflow

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

The project uses a local SQLite MLflow backend.

Start MLflow:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Open:

```text
http://127.0.0.1:5000
```

## 11. FastAPI Inference Service

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

## 12. Test the API

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

## 13. Docker

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

## 14. Automated Testing

The project includes tests for:

- Image preprocessing
- Inference preprocessing
- Prediction utility behavior

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

## 15. Continuous Integration

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

## 16. Docker Hub

The CI workflow publishes the API image to:

```text
DOCKERHUB_USERNAME/cats-dogs-api:latest
```

Docker Hub credentials are stored using GitHub repository secrets and variables.

Required configuration:

```text
Secret:
DOCKERHUB_TOKEN

Variable:
DOCKERHUB_USERNAME
```

## 17. Continuous Deployment

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

## 18. Docker Compose Deployment

Start the full stack manually:

```powershell
docker compose -f deployment\docker-compose.yml up -d
```

Force recreation:

```powershell
docker compose -f deployment\docker-compose.yml up -d --force-recreate --remove-orphans
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

## 19. Smoke Tests

Run:

```powershell
python scripts\smoke_test.py
```

The smoke test validates:

1. `/health` returns HTTP 200 and reports the service as healthy.
2. `/predict` accepts an image and returns the required prediction fields.

The script exits with a non-zero status if either check fails, causing the CD pipeline to fail.

## 20. Monitoring with Prometheus

Prometheus is available at:

```text
http://127.0.0.1:9090
```

The API is scraped from:

```text
http://cats-dogs-api:8000/metrics
```

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

## 21. Grafana

Grafana is available at:

```text
http://127.0.0.1:3000
```

Prometheus data source:

```text
http://prometheus:9090
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

## 22. Application Logging

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

## 23. Post-Deployment Performance Evaluation

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

## 24. Generate Monitoring Traffic

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

## 25. Key URLs

| Service | URL |
|---|---|
| FastAPI | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| Health | `http://127.0.0.1:8000/health` |
| Metrics | `http://127.0.0.1:8000/metrics` |
| MLflow | `http://127.0.0.1:5000` |
| Prometheus | `http://127.0.0.1:9090` |
| Grafana | `http://127.0.0.1:3000` |

## 26. MLOps Pipeline Summary

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

## 27. Final Demonstration Flow

Recommended demonstration sequence:

```text
1. Show GitHub repository and DVC files
2. Show MLflow experiment and metrics
3. Show FastAPI /docs
4. Run pytest
5. Push a code change
6. Show GitHub Actions CI
7. Show Docker Hub image
8. Show GitHub Actions CD
9. Show successful smoke tests
10. Show deployed prediction
11. Show Prometheus target and metrics
12. Show Grafana dashboard
13. Show post-deployment performance results
```

## 28. Deliverables

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
