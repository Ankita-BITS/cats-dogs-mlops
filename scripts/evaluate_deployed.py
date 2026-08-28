import json
import random
from pathlib import Path

import requests

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


BASE_URL = "http://127.0.0.1:8000"

TEST_DIR = Path(
    "data/processed/test"
)

OUTPUT_DIR = Path(
    "artifacts"
)

SAMPLES_PER_CLASS = 20

RANDOM_SEED = 42


def get_images(folder, count):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    images = [
        file
        for file in folder.iterdir()
        if file.suffix.lower()
        in extensions
    ]

    random.shuffle(images)

    return images[:count]


def predict(image_path):

    content_type = "image/jpeg"

    if image_path.suffix.lower() == ".png":
        content_type = "image/png"

    with open(
        image_path,
        "rb"
    ) as image_file:

        files = {
            "file": (
                image_path.name,
                image_file,
                content_type
            )
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            files=files,
            timeout=30
        )

    response.raise_for_status()

    return response.json()


def main():

    random.seed(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cat_images = get_images(
        TEST_DIR / "cats",
        SAMPLES_PER_CLASS
    )

    dog_images = get_images(
        TEST_DIR / "dogs",
        SAMPLES_PER_CLASS
    )

    samples = []

    for image in cat_images:
        samples.append(
            (image, 0, "cat")
        )

    for image in dog_images:
        samples.append(
            (image, 1, "dog")
        )

    random.shuffle(samples)

    true_labels = []
    predicted_labels = []

    prediction_details = []

    print(
        f"Evaluating "
        f"{len(samples)} deployed requests..."
    )

    for number, (
        image_path,
        true_value,
        true_name
    ) in enumerate(
        samples,
        start=1
    ):

        result = predict(
            image_path
        )

        predicted_name = (
            result["label"]
        )

        predicted_value = (
            1
            if predicted_name == "dog"
            else 0
        )

        true_labels.append(
            true_value
        )

        predicted_labels.append(
            predicted_value
        )

        prediction_details.append({
            "filename": image_path.name,
            "true_label": true_name,
            "predicted_label": predicted_name,
            "confidence": result[
                "confidence"
            ]
        })

        print(
            f"{number:02d}/{len(samples)} "
            f"{image_path.name}: "
            f"true={true_name}, "
            f"predicted={predicted_name}"
        )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels
    ).tolist()

    results = {
        "total_requests": len(samples),
        "cats_tested": len(cat_images),
        "dogs_tested": len(dog_images),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix,
        "predictions": prediction_details
    }

    output_path = (
        OUTPUT_DIR /
        "deployed_performance.json"
    )

    with open(
        output_path,
        "w"
    ) as output_file:

        json.dump(
            results,
            output_file,
            indent=4
        )

    print(
        "\n=============================="
    )

    print(
        "DEPLOYED MODEL PERFORMANCE"
    )

    print(
        "=============================="
    )

    print(
        f"Requests:  {len(samples)}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()