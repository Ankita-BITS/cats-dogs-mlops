import sys
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:9999"

TEST_IMAGE = Path(
    "sample_images/cat.jpg"
)


def test_health():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Health check failed: "
            f"{response.status_code}"
        )

    data = response.json()

    if data.get("status") != "healthy":
        raise RuntimeError(
            f"Unexpected health response: "
            f"{data}"
        )

    print("Health check passed")


def test_prediction():

    if not TEST_IMAGE.exists():

        raise RuntimeError(
            f"Smoke test image not found: "
            f"{TEST_IMAGE}"
        )

    with open(TEST_IMAGE, "rb") as image_file:

        files = {
            "file": (
                TEST_IMAGE.name,
                image_file,
                "image/jpeg"
            )
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            files=files,
            timeout=30
        )

    if response.status_code != 200:

        raise RuntimeError(
            f"Prediction failed: "
            f"{response.status_code} "
            f"{response.text}"
        )

    data = response.json()

    required_fields = {
        "label",
        "confidence",
        "cat_probability",
        "dog_probability"
    }

    missing = (
        required_fields -
        set(data.keys())
    )

    if missing:

        raise RuntimeError(
            f"Prediction response missing: "
            f"{missing}"
        )

    print(
        "Prediction smoke test passed"
    )

    print(data)


def main():

    try:

        test_health()
        test_prediction()

        print(
            "\nAll smoke tests passed."
        )

    except Exception as exc:

        print(
            f"\nSmoke test failed: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()