import io
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.inference import load_model, predict_image


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Cats vs Dogs Classification API",
    description=(
        "Binary image classification API "
        "for a pet adoption platform."
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# Load model once when application starts
# ---------------------------------------------------------

model = None
device = None


@app.on_event("startup")
def startup_event():

    global model
    global device

    logger.info("Loading classification model")

    model, device = load_model()

    logger.info(
        "Model loaded successfully on %s",
        device
    )


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Cats vs Dogs Classification API",
        "docs": "/docs"
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "healthy",
        "model": "CatDogCNN"
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    start_time = time.perf_counter()

    allowed_content_types = {
        "image/jpeg",
        "image/png",
        "image/jpg"
    }

    if file.content_type not in allowed_content_types:

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
            time.perf_counter() -
            start_time
        )

        logger.info(
            "Prediction completed "
            "label=%s confidence=%.4f "
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

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process image"
        ) from exc