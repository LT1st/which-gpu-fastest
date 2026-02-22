"""Base model interface"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn


class BaseModel(ABC):
    """Abstract base class for benchmark models"""

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the PyTorch model"""
        pass

    @abstractmethod
    def get_input_fn(self) -> Callable:
        """Return a function that generates input for the model"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name"""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type (cnn, transformer, etc.)"""
        pass


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
