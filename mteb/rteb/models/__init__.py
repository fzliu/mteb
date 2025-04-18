from __future__ import annotations

from ..core.base.model import EmbeddingModel
from ..core.meta import ModelMeta, model_id  # Use local ebr ModelMeta
from ..utils.lazy_import import LazyImport
from .bgem3 import *
from .cohere import *
from .google import *
from .gritlm import *
from .openai import *
from .sentence_transformers import *
from .voyageai import *

MODEL_REGISTRY: dict[str, ModelMeta] = {}
for name in dir():
    meta = eval(name)
    # Explicitly exclude `LazyImport` instances since the latter check invokes the import.
    if not isinstance(meta, LazyImport) and isinstance(meta, ModelMeta):
        MODEL_REGISTRY[meta._id] = eval(name)


def get_embedding_model(
    model_name: str, embd_dim: int, embd_dtype: str, **kwargs
) -> EmbeddingModel:
    key = model_id(model_name, embd_dim, embd_dtype)
    # TODO: add logic to dynamically load missing model
    return MODEL_REGISTRY[key].load_model(**kwargs)
