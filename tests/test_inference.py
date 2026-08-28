import torch

from PIL import Image

from src.inference import (
    preprocess_image,
    predict_image
)


class FakeDogModel(torch.nn.Module):
    """
    Fake model that always produces a
    positive logit, resulting in a dog
    prediction.
    """

    def forward(self, x):

        batch_size = x.shape[0]

        return torch.full(
            (batch_size, 1),
            2.0
        )


def test_inference_preprocessing():

    image = Image.new(
        "RGB",
        (500, 400)
    )

    tensor = preprocess_image(image)

    assert tensor.shape == (
        1,
        3,
        224,
        224
    )


def test_predict_image():

    model = FakeDogModel()

    device = torch.device("cpu")

    image = Image.new(
        "RGB",
        (224, 224)
    )

    result = predict_image(
        model,
        device,
        image
    )

    assert result["label"] == "dog"

    assert (
        result["dog_probability"]
        > result["cat_probability"]
    )

    assert 0 <= result["confidence"] <= 1