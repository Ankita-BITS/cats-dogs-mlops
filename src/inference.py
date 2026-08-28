from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import CatDogCNN


MODEL_PATH = Path("models/model.pt")

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "cat",
    1: "dog"
}


inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(model_path=MODEL_PATH):
    """
    Load the trained Cats vs Dogs CNN model.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CatDogCNN()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image):
    """
    Convert image to RGB, resize and normalize
    exactly as done during model evaluation.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")

    image = image.convert("RGB")

    tensor = inference_transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor


def predict_image(model, device, image):
    """
    Return predicted class and probabilities.
    """

    tensor = preprocess_image(image)

    tensor = tensor.to(device)

    with torch.no_grad():

        logits = model(tensor)

        dog_probability = torch.sigmoid(
            logits
        ).item()

    cat_probability = 1.0 - dog_probability

    predicted_index = (
        1 if dog_probability >= 0.5 else 0
    )

    predicted_label = CLASS_NAMES[
        predicted_index
    ]

    confidence = max(
        cat_probability,
        dog_probability
    )

    return {
        "label": predicted_label,
        "confidence": round(confidence, 4),
        "cat_probability": round(
            cat_probability,
            4
        ),
        "dog_probability": round(
            dog_probability,
            4
        )
    }