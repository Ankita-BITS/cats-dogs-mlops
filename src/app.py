import io
import logging
import time

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from src.inference import load_model, predict_image


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)


# ============================================================
# PROMETHEUS METRICS
# ============================================================

API_REQUESTS = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint"]
)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total failed prediction requests"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Cats vs Dogs Classification API",
    description=(
        "Binary image classification API "
        "for a pet adoption platform."
    ),
    version="1.0.0"
)


# ============================================================
# MODEL
# ============================================================

model = None
device = None


@app.on_event("startup")
def startup_event():

    global model
    global device

    logger.info(
        "Loading Cats vs Dogs classification model"
    )

    model, device = load_model()

    logger.info(
        "Model loaded successfully on %s",
        device
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    API_REQUESTS.labels(
        endpoint="/"
    ).inc()

    return {
        "message": "Cats vs Dogs Classification API",
        "docs": "/docs"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    API_REQUESTS.labels(
        endpoint="/health"
    ).inc()

    if model is None:

        logger.error(
            "Health check failed: model not loaded"
        )

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "healthy",
        "model": "CatDogCNN"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    start_time = time.perf_counter()

    API_REQUESTS.labels(
        endpoint="/predict"
    ).inc()

    PREDICTION_REQUESTS.inc()

    allowed_content_types = {
        "image/jpeg",
        "image/png",
        "image/jpg"
    }

    if file.content_type not in allowed_content_types:

        PREDICTION_ERRORS.inc()

        logger.warning(
            "Prediction rejected: invalid file type %s",
            file.content_type
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Upload a JPEG or PNG image."
            )
        )

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        result = predict_image(
            model,
            device,
            image
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        PREDICTION_LATENCY.observe(
            latency
        )

        logger.info(
            "Prediction completed "
            "label=%s "
            "confidence=%.4f "
            "latency=%.4fs",
            result["label"],
            result["confidence"],
            latency
        )

        return {
            **result,
            "filename": file.filename
        }

    except HTTPException:

        raise

    except Exception as exc:

        PREDICTION_ERRORS.inc()

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process image"
        ) from exc


# ============================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )