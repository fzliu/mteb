from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed

from mteb.encoder_interface import Encoder as MTEBEncoder
from mteb.rteb.utils.data import JSONLDataset
from mteb.rteb.utils.distributed import gather_list

logger = logging.getLogger(__name__)


class MTEBToRTEBEncoderWrapper(pl.LightningModule):
    """Acts as a PyTorch Lightning Module to wrap an MTEB Encoder,
    replicating the necessary functionality of RTEB's Encoder class
    for use with trainer.predict, but overriding __setattr__ to prevent recursion.
    """

    def __init__(
        self,
        mteb_model: MTEBEncoder,
        model_name: str = "mteb_wrapped_model",
        save_embds: bool = False,  # Replicate args from RtebEncoder
        load_embds: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.mteb_model_instance = mteb_model
        self.model_name = model_name
        self._id = model_name  # Used for save paths
        self.query_instruct = ""  # Add instructions if applicable
        self.corpus_instruct = ""  # Add instructions if applicable
        self.embd_dim = None
        self.embd_dtype = "float32"

        # Replicate state/config
        self._load_embds = load_embds
        self._save_embds = save_embds
        self.in_memory = True
        self.is_query = False
        self.save_file = None

        # Internal state
        self.embds = None
        self.local_embds = []
        self.local_existing_ids = set()
        self.local_embd_file = None
        self._private_trainer = None  # Initialize private trainer attribute

    def __setattr__(self, name: str, value: Any) -> None:
        # Override to prevent recursion when Lightning sets the trainer property
        if name == "trainer":
            # Store trainer privately AND *do not* call super().__setattr__ for 'trainer'
            # This prevents the LightningModule's property setter recursion
            # Use object.__setattr__ to bypass the overridden __setattr__ for this specific case
            object.__setattr__(self, "_private_trainer", value)
        else:
            # For all other attributes, use the default LightningModule behavior
            super().__setattr__(name, value)

    # --- Properties expected by run_retrieve_task ---
    @property
    def model(self):
        # Return self to allow access like encoder.model._id -> encoder._id
        # This avoids exposing the mteb_model_instance directly via this property,
        # potentially mitigating the recursion issue, while satisfying attribute access.
        return self

    @property
    def load_embds(self) -> bool:
        return self._load_embds

    @property
    def save_embds(self) -> bool:
        return self._save_embds or not self.in_memory

    @property
    def local_embd_file_name(self) -> str:
        assert self.save_file is not None
        # Ensure trainer and local_rank are available
        # Use the _private_trainer we stored manually
        trainer_instance = getattr(self, "_private_trainer", None)
        num_shards = (
            getattr(trainer_instance, "num_devices", 1) if trainer_instance else 1
        )
        local_rank = getattr(self, "local_rank", 0)
        return f"{self.save_file}-{local_rank}-of-{num_shards}"

    def get_local_embd_files(self, num_shards=None) -> list[str]:
        assert self.save_file is not None
        if num_shards is None:
            trainer_instance = getattr(self, "_private_trainer", None)
            num_shards = (
                getattr(trainer_instance, "num_devices", 1) if trainer_instance else 1
            )
        return [f"{self.save_file}-{i}-of-{num_shards}" for i in range(num_shards)]

    def get_embd_files(self, num_shards=None) -> list[str]:
        local_files = self.get_local_embd_files(num_shards=num_shards)
        return local_files

    def embd_files_exist(self, num_shards=None) -> bool:
        files = self.get_embd_files(num_shards=num_shards)
        return all(os.path.exists(file) for file in files)

    # --- End Properties ---

    def encode(self, sentences: list[str], **kwargs) -> torch.Tensor:
        """Encodes sentences using the wrapped MTEB model and returns torch.Tensor."""
        embeddings = self.mteb_model_instance.encode(sentences, **kwargs)
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
    def on_predict_epoch_start(self):
        self.embds = None
        if self.in_memory:
            self.local_embds = []

        if self.load_embds:
            self.local_existing_ids = set()
            file_path = self.local_embd_file_name if self.save_file else None
            if file_path and os.path.exists(file_path):
                logger.warning(f"Load embeddings from {file_path}")
                try:
                    ds = JSONLDataset(file_path)
                    for example in ds:
                        self.local_existing_ids.add(example["id"])
                        if self.in_memory:
                            self.local_embds.append(example)
                except Exception as e:
                    logger.error(f"Failed to load embeddings from {file_path}: {e}")
                    self.local_existing_ids = set()
                    self.local_embds = []
            elif self.load_embds:
                logger.warning(
                    f"load_embds is True but {file_path} doesn't exist. Skipping loading."
                )

        if self.save_embds:
            file_path = self.local_embd_file_name if self.save_file else None
            if file_path:
                mode = "a" if self.load_embds and os.path.exists(file_path) else "w"
                try:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    self.local_embd_file = open(file_path, mode)
                except Exception as e:
                    logger.error(
                        f"Failed to open embedding file {file_path} in mode '{mode}': {e}"
                    )
                    self.local_embd_file = None
            else:
                logger.warning(
                    "save_embds is True, but save_file is not set. Cannot save embeddings."
                )
                self.local_embd_file = None

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, dict) or "id" not in batch or "sentences" not in batch:
            logger.error(
                f"Unsupported batch type or missing keys in predict_step: {type(batch)}"
            )
            return

        indices = batch["id"]
        sentences = batch["sentences"]

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
            # Pass task_name from self.model_name (which was set during init)
            embds = self.encode(sentences, task_name=self.model_name)
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

    def on_predict_epoch_end(self):
        if self.save_embds and self.local_embd_file:
            try:
                self.local_embd_file.close()
            except Exception as e:
                logger.error(
                    f"Failed to close embedding file {self.local_embd_file_name}: {e}"
                )
            self.local_embd_file = None

        if self.in_memory:
            trainer_instance = getattr(self, "_private_trainer", None)
            num_devices = (
                getattr(trainer_instance, "num_devices", 1) if trainer_instance else 1
            )
            # Only gather if multiple devices were used
            if num_devices > 1:
                try:
                    if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                    ):
                        self.embds = gather_list(self.local_embds, num_devices)
                    else:
                        logger.warning(
                            "Distributed environment not available/initialized, cannot gather embeddings."
                        )
                        self.embds = self.local_embds
                except Exception as e:
                    logger.error(f"Failed to gather embeddings: {e}")
                    self.embds = self.local_embds

        trainer_instance = getattr(self, "_private_trainer", None)
        if (
            trainer_instance
            and hasattr(trainer_instance, "strategy")
            and hasattr(trainer_instance.strategy, "barrier")
        ):
            try:
                # Use the stored trainer instance
                trainer_instance.strategy.barrier()
            except Exception as e:
                logger.error(f"Failed to execute barrier: {e}")

    def apply(self, fn):
        # Override apply to prevent recursion into the wrapped mteb_model_instance
        super().apply(fn)
        return self

    # --- End Replicated Hooks ---
