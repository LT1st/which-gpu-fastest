"""CNN model implementations"""

from typing import Callable, Tuple, Any

import torch
import torch.nn as nn

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# CNN model configurations
CNN_MODELS = {
    "resnet18": {
        "name": "ResNet-18",
        "params": "11.7M",
        "input_size": (224, 224),
        "description": "Lightweight residual network",
    },
    "resnet34": {
        "name": "ResNet-34",
        "params": "21.8M",
        "input_size": (224, 224),
        "description": "Medium residual network",
    },
    "resnet50": {
        "name": "ResNet-50",
        "params": "25.6M",
        "input_size": (224, 224),
        "description": "Standard residual network for image classification",
    },
    "resnet101": {
        "name": "ResNet-101",
        "params": "44.5M",
        "input_size": (224, 224),
        "description": "Deep residual network",
    },
    "resnet152": {
        "name": "ResNet-152",
        "params": "60.2M",
        "input_size": (224, 224),
        "description": "Very deep residual network",
    },
    "vgg16": {
        "name": "VGG-16",
        "params": "138M",
        "input_size": (224, 224),
        "description": "VGG network with 16 layers",
    },
    "vgg19": {
        "name": "VGG-19",
        "params": "144M",
        "input_size": (224, 224),
        "description": "VGG network with 19 layers",
    },
    "alexnet": {
        "name": "AlexNet",
        "params": "61M",
        "input_size": (224, 224),
        "description": "Classic AlexNet architecture",
    },
    "mobilenet_v2": {
        "name": "MobileNet V2",
        "params": "3.5M",
        "input_size": (224, 224),
        "description": "Efficient mobile architecture",
    },
    "mobilenet_v3_small": {
        "name": "MobileNet V3 Small",
        "params": "2.5M",
        "input_size": (224, 224),
        "description": "Lightweight mobile architecture",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "params": "5.3M",
        "input_size": (224, 224),
        "description": "EfficientNet baseline",
    },
    "densenet121": {
        "name": "DenseNet-121",
        "params": "8M",
        "input_size": (224, 224),
        "description": "Dense connectivity network",
    },
}


def get_cnn_model(model_name: str) -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """
    Get a CNN model by name.

    Args:
        model_name: Name of the CNN model

    Returns:
        Tuple of (model, input_shape, get_input_fn)
    """
    # Handle simple_cnn separately (doesn't require torchvision)
    if model_name == "simple_cnn":
        return get_simple_cnn()

    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required for CNN models")

    # Get model constructor
    model_constructors = {
        "resnet18": tv_models.resnet18,
        "resnet34": tv_models.resnet34,
        "resnet50": tv_models.resnet50,
        "resnet101": tv_models.resnet101,
        "resnet152": tv_models.resnet152,
        "vgg16": tv_models.vgg16,
        "vgg19": tv_models.vgg19,
        "alexnet": tv_models.alexnet,
        "mobilenet_v2": tv_models.mobilenet_v2,
        "mobilenet_v3_small": tv_models.mobilenet_v3_small,
        "efficientnet_b0": tv_models.efficientnet_b0,
        "densenet121": tv_models.densenet121,
    }

    if model_name not in model_constructors:
        raise ValueError(f"Unknown CNN model: {model_name}")

    # Create model without pretrained weights
    model = model_constructors[model_name](weights=None)

    # Get input size
    config = CNN_MODELS.get(model_name, {})
    input_size = config.get("input_size", (224, 224))

    # Input shape: (batch_size, channels, height, width)
    input_shape = (1, 3, input_size[0], input_size[1])

    def get_input_fn(batch_size: int, device: str):
        """Generate random input for CNN"""
        return torch.randn(
            batch_size, 3, input_size[0], input_size[1],
            device=device, dtype=torch.float32
        )

    return model, input_shape, get_input_fn


# Custom simple CNN for basic testing
class SimpleCNN(nn.Module):
    """Simple CNN for basic testing"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Add simple CNN to models
CNN_MODELS["simple_cnn"] = {
    "name": "Simple CNN",
    "params": "~5M",
    "input_size": (224, 224),
    "description": "Simple 3-layer CNN for basic testing",
}


def get_simple_cnn():
    """Get SimpleCNN model"""
    model = SimpleCNN(num_classes=10)
    input_shape = (1, 3, 224, 224)

    def get_input_fn(batch_size: int, device: str):
        return torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32)

    return model, input_shape, get_input_fn
