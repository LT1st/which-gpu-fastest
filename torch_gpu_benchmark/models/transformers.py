"""Transformer model implementations"""

from typing import Callable, Tuple, Any

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Transformer model configurations
TRANSFORMER_MODELS = {
    "bert-tiny": {
        "name": "BERT-Tiny",
        "params": "4.4M",
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 2,
        "seq_length": 128,
        "description": "Tiny BERT for fast testing",
    },
    "bert-mini": {
        "name": "BERT-Mini",
        "params": "11.3M",
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "seq_length": 128,
        "description": "Mini BERT model",
    },
    "bert-small": {
        "name": "BERT-Small",
        "params": "29M",
        "hidden_size": 512,
        "num_layers": 4,
        "num_heads": 8,
        "seq_length": 128,
        "description": "Small BERT model",
    },
    "bert-base": {
        "name": "BERT-Base",
        "params": "110M",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "seq_length": 128,
        "description": "Standard BERT base model",
    },
    "gpt2-small": {
        "name": "GPT-2 Small",
        "params": "124M",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "seq_length": 128,
        "description": "GPT-2 small model",
    },
    "distilbert": {
        "name": "DistilBERT",
        "params": "66M",
        "hidden_size": 768,
        "num_layers": 6,
        "num_heads": 12,
        "seq_length": 128,
        "description": "Distilled BERT model",
    },
}


def get_transformer_model(model_name: str) -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """
    Get a transformer model by name.

    Args:
        model_name: Name of the transformer model

    Returns:
        Tuple of (model, input_shape, get_input_fn)
    """
    if model_name in ["bert-tiny", "bert-mini", "bert-small", "bert-base"]:
        return get_bert_model(model_name)
    elif model_name == "gpt2-small":
        return get_gpt2_model()
    elif model_name == "distilbert":
        return get_distilbert_model()
    else:
        # Try to load from HuggingFace
        return get_hf_model(model_name)


def get_bert_model(model_name: str) -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """Get BERT model"""
    # Map to HuggingFace model names
    hf_names = {
        "bert-tiny": "prajjwal1/bert-tiny",
        "bert-mini": "prajjwal1/bert-mini",
        "bert-small": "prajjwal1/bert-small",
        "bert-base": "bert-base-uncased",
    }

    config = TRANSFORMER_MODELS.get(model_name, TRANSFORMER_MODELS["bert-tiny"])
    seq_length = config["seq_length"]

    if HAS_TRANSFORMERS:
        try:
            hf_name = hf_names.get(model_name, "prajjwal1/bert-tiny")
            model = AutoModel.from_pretrained(hf_name)
            vocab_size = model.config.vocab_size
        except Exception:
            model = create_simple_bert(config)
            vocab_size = 30522
    else:
        model = create_simple_bert(config)
        vocab_size = 30522

    input_shape = (1, seq_length)

    def get_input_fn(batch_size: int, device: str):
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
            "attention_mask": torch.ones(batch_size, seq_length, device=device, dtype=torch.long),
        }

    return model, input_shape, get_input_fn


def get_gpt2_model() -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """Get GPT-2 small model"""
    config = TRANSFORMER_MODELS["gpt2-small"]
    seq_length = config["seq_length"]

    if HAS_TRANSFORMERS:
        try:
            from transformers import GPT2Model
            model = GPT2Model.from_pretrained("gpt2")
            vocab_size = model.config.vocab_size
        except Exception:
            model = create_simple_gpt2(config)
            vocab_size = 50257
    else:
        model = create_simple_gpt2(config)
        vocab_size = 50257

    input_shape = (1, seq_length)

    def get_input_fn(batch_size: int, device: str):
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
            "attention_mask": torch.ones(batch_size, seq_length, device=device, dtype=torch.long),
        }

    return model, input_shape, get_input_fn


def get_distilbert_model() -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """Get DistilBERT model"""
    config = TRANSFORMER_MODELS["distilbert"]
    seq_length = config["seq_length"]

    if HAS_TRANSFORMERS:
        try:
            from transformers import DistilBertModel
            model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            vocab_size = model.config.vocab_size
        except Exception:
            model = create_simple_bert(config)
            vocab_size = 30522
    else:
        model = create_simple_bert(config)
        vocab_size = 30522

    input_shape = (1, seq_length)

    def get_input_fn(batch_size: int, device: str):
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
            "attention_mask": torch.ones(batch_size, seq_length, device=device, dtype=torch.long),
        }

    return model, input_shape, get_input_fn


def get_hf_model(model_name: str) -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """Load any HuggingFace model"""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library required for HuggingFace models")

    model = AutoModel.from_pretrained(model_name)
    seq_length = 128
    vocab_size = getattr(model.config, "vocab_size", 30522)

    input_shape = (1, seq_length)

    def get_input_fn(batch_size: int, device: str):
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
            "attention_mask": torch.ones(batch_size, seq_length, device=device, dtype=torch.long),
        }

    return model, input_shape, get_input_fn


# Simple BERT implementation for testing without HuggingFace
class SimpleBERT(nn.Module):
    """Simple BERT-like model for testing"""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
    ):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
        })()

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        embeddings = self.embeddings(input_ids) + self.position_embeddings(position_ids)

        if attention_mask is not None:
            # Convert to attention mask format
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=embeddings.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = self.encoder(embeddings, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)

        return {"last_hidden_state": hidden_states}


def create_simple_bert(config: dict) -> SimpleBERT:
    """Create SimpleBERT from config"""
    return SimpleBERT(
        hidden_size=config.get("hidden_size", 768),
        num_layers=config.get("num_layers", 12),
        num_heads=config.get("num_heads", 12),
    )


class SimpleGPT2(nn.Module):
    """Simple GPT-2-like model for testing"""

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_position_embeddings: int = 1024,
    ):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
        })()

        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.wpe = nn.Embedding(max_position_embeddings, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.encoder(hidden_states)

        return {"last_hidden_state": hidden_states}


def create_simple_gpt2(config: dict) -> SimpleGPT2:
    """Create SimpleGPT2 from config"""
    return SimpleGPT2(
        hidden_size=config.get("hidden_size", 768),
        num_layers=config.get("num_layers", 12),
        num_heads=config.get("num_heads", 12),
    )
