"""Model definitions and factories"""

from typing import Dict, Callable, Tuple, Any
from .base import BaseModel
from .cnn import get_cnn_model, CNN_MODELS
from .transformers import get_transformer_model, TRANSFORMER_MODELS
from .generative import get_generative_model, GENERATIVE_MODELS

# All available models
ALL_MODELS = {
    **CNN_MODELS,
    **TRANSFORMER_MODELS,
    **GENERATIVE_MODELS,
}


def get_model_factory(model_name: str) -> Tuple[Callable, str]:
    """
    Get model factory and type for a given model name.

    Args:
        model_name: Name of the model

    Returns:
        Tuple of (factory_function, model_type)

    Raises:
        ValueError: If model name is not recognized
    """
    if model_name in CNN_MODELS:
        return get_cnn_model, "cnn"
    elif model_name in TRANSFORMER_MODELS:
        return get_transformer_model, "transformer"
    elif model_name in GENERATIVE_MODELS:
        return get_generative_model, "generative"
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(ALL_MODELS.keys())}")


def list_models(model_type: str = None) -> Dict[str, Dict[str, Any]]:
    """
    List available models.

    Args:
        model_type: Filter by type (cnn, transformer, generative). None for all.

    Returns:
        Dictionary of model info
    """
    if model_type == "cnn":
        return CNN_MODELS
    elif model_type == "transformer":
        return TRANSFORMER_MODELS
    elif model_type == "generative":
        return GENERATIVE_MODELS
    else:
        return ALL_MODELS
