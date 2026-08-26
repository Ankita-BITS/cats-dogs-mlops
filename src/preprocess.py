from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import random

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42


def preprocess_image(input_path, output_path):
    """
    Convert image to RGB, resize to 224x224,
    and save to the processed dataset.
    """
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image.save(output_path)


def create_directories():
    """Create train/validation/test directories."""

    for split in ["train", "val", "test"]:
        for label in ["cats", "dogs"]:
            (PROCESSED_DIR / split / label).mkdir(
                parents=True,
                exist_ok=True
            )


def split_files(files):
    """
    Split data into:
    80% training
    10% validation
    10% testing
    """

    train_files, temp_files = train_test_split(
        files,
        test_size=0.20,
        random_state=RANDOM_SEED
    )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=RANDOM_SEED
    )

    return train_files, val_files, test_files


def process_split(files, split, label):
    """Preprocess and save images for one dataset split."""

    output_dir = PROCESSED_DIR / split / label

    processed = 0
    skipped = 0

    for file_path in files:

        output_path = output_dir / file_path.name

        try:
            preprocess_image(file_path, output_path)
            processed += 1

        except Exception as exc:
            skipped += 1
            print(f"Skipping {file_path}: {exc}")

    print(
        f"{label} - {split}: "
        f"{processed} processed, {skipped} skipped"
    )


def main():

    random.seed(RANDOM_SEED)

    create_directories()

    # Actual Kaggle dataset folder structure
    class_folders = {
        "cats": RAW_DIR / "Cat",
        "dogs": RAW_DIR / "Dog",
    }

    for label, source_dir in class_folders.items():

        files = list(source_dir.glob("*.jpg"))
        files += list(source_dir.glob("*.jpeg"))
        files += list(source_dir.glob("*.png"))

        print()
        print(f"{label}: {len(files)} images found in {source_dir}")

        if len(files) == 0:
            raise ValueError(
                f"No images found in {source_dir}. "
                f"Check the dataset folder structure."
            )

        train_files, val_files, test_files = split_files(files)

        print(
            f"{label}: "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )

        process_split(
            train_files,
            "train",
            label
        )

        process_split(
            val_files,
            "val",
            label
        )

        process_split(
            test_files,
            "test",
            label
        )

    print()
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()