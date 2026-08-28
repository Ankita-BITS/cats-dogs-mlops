from PIL import Image

from src.preprocess import preprocess_image


def test_preprocess_image(tmp_path):
    """
    Test that preprocessing:
    1. Converts image to RGB
    2. Resizes image to 224x224
    3. Saves the processed image
    """

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.jpg"

    # Create a temporary grayscale image
    image = Image.new(
        mode="L",
        size=(500, 300),
        color=128
    )

    image.save(input_path)

    preprocess_image(
        input_path,
        output_path
    )

    assert output_path.exists()

    processed_image = Image.open(
        output_path
    )

    assert processed_image.size == (
        224,
        224
    )

    assert processed_image.mode == "RGB"