"""
Smoke tests: verify all core dependencies import cleanly.
These run in CI without any dataset or GPU.
"""


def test_torch_imports():
    import torch
    import torchvision

    assert torch.__version__, "torch version should be non-empty"
    assert torchvision.__version__, "torchvision version should be non-empty"


def test_timm_import():
    import timm

    assert timm.__version__, "timm version should be non-empty"


def test_image_libs():
    import cv2
    from PIL import Image
    import numpy as np

    # Basic sanity: create a tiny numpy array and wrap it as a PIL image
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    assert img.size == (4, 4)


def test_data_libs():
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend, safe in CI
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert list(df.columns) == ["a", "b"]


def test_torch_cpu_tensor():
    import torch

    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0


def test_resnet_model_instantiation():
    """Verify a ResNet can be instantiated (no dataset required)."""
    import torchvision.models as models

    model = models.resnet18(pretrained=False)
    assert model is not None


def test_timm_swin_instantiation():
    """Verify a Swin Transformer variant can be instantiated via timm."""
    import timm

    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    assert model is not None
