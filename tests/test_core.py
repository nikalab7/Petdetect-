from PIL import Image
import torch

from core import make_model, make_transform, predict_image


def test_transforms_output_correct_shape():
    image = Image.new('RGB', (300, 260), color='red')
    assert tuple(make_transform(224)(image).shape) == (3, 224, 224)
    assert tuple(make_transform(160)(image).shape) == (3, 160, 160)


def test_model_predicts_all_37_logits_without_pretrained_download():
    model = make_model(pretrained=False).eval()
    with torch.inference_mode():
        assert tuple(model(torch.zeros(1, 3, 160, 160)).shape) == (1, 37)


def test_prediction_top_three_is_ordered():
    image = Image.new('RGB', (300, 300), color='white')
    model = make_model(pretrained=False).eval()
    results = predict_image(model, image, 160, torch.device('cpu'), [str(i) for i in range(37)])
    assert len(results) == 3
    assert all(0 <= p <= 1 for _, p in results)
    assert results[0][1] >= results[1][1] >= results[2][1]
