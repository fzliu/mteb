from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch
import torch.distributed

from mteb.encoder_interface import Encoder as MTEBEncoder
from mteb.rteb.core.encoder import Encoder as RTEBEncoder

logger = logging.getLogger(__name__)


class MTEBToRTEBEncoderWrapper(RTEBEncoder):
    """Acts as a PyTorch Lightning Module to wrap an MTEB Encoder,
    replicating the necessary functionality of RTEB's Encoder class
    for use with trainer.predict, but overriding __setattr__ to prevent recursion.
    """

    def __init__(
        self,
        mteb_model: MTEBEncoder,
        task_name: str,
        model_name: str = "mteb_wrapped_model",
        save_embds: bool = False,
        load_embds: bool = False,
        batch_size: int = 16,
        **kwargs,
    ):
        super().__init__(None, save_embds, load_embds, **kwargs)
        self.mteb_model_instance = mteb_model
        self.model_name = model_name
        self.task_name = task_name
        self.batch_size = batch_size
        self.query_instruct = ""  # Add instructions if applicable
        self.corpus_instruct = ""  # Add instructions if applicable
        self.embd_dim = None
        self.embd_dtype = "float32"

        # Internal state
        self.embds = None
        self.local_embds = []
        self.local_existing_ids = set()
        self.local_embd_file = None

    # --- Properties expected by run_retrieve_task ---
    @property
    def model(self):
        return self

    # --- End Properties ---

    def encode(self, sentences: list[str], **kwargs) -> torch.Tensor:
        """Encodes sentences using the wrapped MTEB model and returns torch.Tensor."""
        embeddings = self.mteb_model_instance.encode(
            sentences, batch_size=self.batch_size, **kwargs
        )
        if self.embd_dim is None and hasattr(embeddings, "shape"):
            if len(embeddings.shape) >= 2:
                self.embd_dim = embeddings.shape[1]
            elif len(embeddings.shape) == 1 and embeddings.shape[0] == 0:
                pass
            else:
                logger.warning(
                    f"Unexpected embedding shape: {embeddings.shape}. Cannot determine embd_dim."
                )

        if isinstance(embeddings, np.ndarray):
            return torch.from_numpy(embeddings).to(torch.float32)
        elif isinstance(embeddings, torch.Tensor):
            return embeddings.to(torch.float32)
        elif isinstance(embeddings, list):
            if not embeddings:
                dim = self.embd_dim if self.embd_dim is not None else 768
                return torch.empty((0, dim), dtype=torch.float32)
            if isinstance(embeddings[0], np.ndarray):
                return torch.from_numpy(np.stack(embeddings)).to(torch.float32)
            elif isinstance(embeddings[0], torch.Tensor):
                return torch.stack(embeddings).to(torch.float32)
            else:
                raise TypeError(
                    f"Unsupported embedding list element type: {type(embeddings[0])}"
                )
        else:
            raise TypeError(
                f"Unsupported embedding type from MTEB model: {type(embeddings)}"
            )

    # --- Replicated predict hooks from RtebEncoder ---
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, dict) or "id" not in batch or "text" not in batch:
            logger.error(
                f"Unsupported batch type or missing keys in predict_step: {type(batch)}"
            )
            return

        indices = batch["id"]
        sentences = batch["text"]

        if not indices or not sentences:
            return

        if self.load_embds and self.local_existing_ids:
            if all(idx in self.local_existing_ids for idx in indices):
                return
            if any(idx in self.local_existing_ids for idx in indices):
                logger.warning(
                    "Partial loading within batch detected, but not supported. Re-encoding entire batch."
                )

        try:
            embds = self.encode(sentences, task_name=self.task_name)
        except Exception as e:
            logger.error(
                f"Encoding failed for batch_idx {batch_idx}: {e}", exc_info=True
            )
            return

        for idx, embd in zip(indices, embds):
            embd_list = embd.tolist()
            obj = {"id": idx, "embd": embd_list}

            if self.in_memory:
                if not (self.load_embds and idx in self.local_existing_ids):
                    self.local_embds.append(obj)

            if self.save_embds and self.local_embd_file:
                if not (self.load_embds and idx in self.local_existing_ids):
                    try:
                        self.local_embd_file.write(json.dumps(obj) + "\n")
                    except Exception as e:
                        logger.error(
                            f"Failed to write embedding for ID {idx} to file: {e}"
                        )

    def apply(self, fn):
        # Override apply to prevent recursion into the wrapped mteb_model_instance
        super().apply(fn)
        return self

    # --- End Replicated Hooks ---
