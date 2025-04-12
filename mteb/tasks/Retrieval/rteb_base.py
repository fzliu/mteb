# Base class and wrapper for RTEB task integration
from __future__ import annotations

import argparse
import logging
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

# MTEB Imports
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import HFSubset, TaskMetadata
from mteb.encoder_interface import Encoder as MTEBEncoder
from mteb.load_results.task_results import ScoresDict

# RTEB Imports
from mteb.rteb.ebr.core.encoder import Encoder as RtebEncoder
from mteb.rteb.ebr.core.meta import DatasetMeta
from mteb.rteb.ebr.core.retriever import Retriever
from mteb.rteb.ebr.retrieve import run_retrieve_task

logger = logging.getLogger(__name__)


# --- RTEB Encoder Wrapper ---
class MTEBToRTEBEncoderWrapper(RtebEncoder):
    """Wraps an MTEB Encoder to be compatible with RTEB's Encoder interface."""

    def __init__(self, mteb_model: MTEBEncoder, model_name: str = "mteb_wrapped_model"):
        # Note: RtebEncoder's __init__ might take arguments, adjust if needed.
        # Calling parent __init__ might be necessary depending on RtebEncoder implementation.
        # super().__init__() # Uncomment if RtebEncoder requires initialization
        self.model = mteb_model
        # RTEB's Encoder might expect these attributes, adjust as needed
        self.model_name = model_name
        self._id = model_name  # Used for save paths in RTEB
        self.query_instruct = ""  # Add instructions if applicable
        self.corpus_instruct = ""  # Add instructions if applicable
        self.embd_dim = None  # Will be set after first encode
        self.embd_dtype = "float32"  # Assuming float32

        # Required attributes from pl.LightningModule which RtebEncoder likely inherits
        self._trainer = None
        self._current_fx_name = None

    def forward(self, **kwargs) -> Any:
        # This might not be directly used if RTEB calls encode directly
        raise NotImplementedError("Forward not implemented for wrapper.")

    def encode(self, sentences: list[str], **kwargs) -> torch.Tensor:
        """Encodes sentences using the wrapped MTEB model and returns torch.Tensor."""
        embeddings = self.model.encode(sentences, **kwargs)
        if self.embd_dim is None and hasattr(embeddings, "shape"):
            # Check if shape is valid (at least 2 dimensions)
            if len(embeddings.shape) >= 2:
                self.embd_dim = embeddings.shape[1]
            elif (
                len(embeddings.shape) == 1 and embeddings.shape[0] == 0
            ):  # Handle empty case
                pass  # embd_dim remains None, handle downstream or set default
            else:
                logger.warning(
                    f"Unexpected embedding shape: {embeddings.shape}. Cannot determine embd_dim."
                )

        # Ensure output is torch.Tensor
        if isinstance(embeddings, np.ndarray):
            return torch.from_numpy(embeddings)
        elif isinstance(embeddings, torch.Tensor):
            return embeddings
        elif isinstance(
            embeddings, list
        ):  # Handle list of tensors/arrays if model returns that
            if not embeddings:
                # Use a reasonable default dimension if embd_dim wasn't set
                dim = self.embd_dim if self.embd_dim is not None else 768
                return torch.empty((0, dim), dtype=torch.float32)  # Handle empty list
            if isinstance(embeddings[0], np.ndarray):
                return torch.from_numpy(np.stack(embeddings))
            elif isinstance(embeddings[0], torch.Tensor):
                return torch.stack(embeddings)
            else:
                raise TypeError(
                    f"Unsupported embedding list element type: {type(embeddings[0])}"
                )
        else:
            raise TypeError(
                f"Unsupported embedding type from MTEB model: {type(embeddings)}"
            )

    # Add dummy implementations for methods potentially required by pl.Trainer predict hooks
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # This method is called by trainer.predict.
        # It should call the encode method. The exact batch structure depends
        # on how RetrieveDataModule yields data. Assuming it yields dicts with 'sentences'.
        if isinstance(batch, dict) and "sentences" in batch:
            # Handle potential empty batch from dataloader
            if not batch["sentences"]:
                # Use a reasonable default dimension if embd_dim wasn't set
                dim = self.embd_dim if self.embd_dim is not None else 768
                return torch.empty((0, dim), dtype=torch.float32)
            return self.encode(batch["sentences"])
        elif isinstance(batch, list):  # Assuming batch is just a list of sentences
            if not batch:
                # Use a reasonable default dimension if embd_dim wasn't set
                dim = self.embd_dim if self.embd_dim is not None else 768
                return torch.empty((0, dim), dtype=torch.float32)
            return self.encode(batch)
        else:
            raise TypeError(f"Unsupported batch type in predict_step: {type(batch)}")

    # Potentially add other methods required by RtebEncoder or pl.LightningModule if any


# --- End RTEB Encoder Wrapper ---


# --- Base Class for RTEB Tasks ---
class AbsTaskRTEBRetrieval(AbsTaskRetrieval, ABC):  # Explicitly mark as abstract
    """Abstract base class for integrating RTEB retrieval tasks into MTEB."""

    # Subclasses MUST define these
    metadata: TaskMetadata
    rteb_data_path: str
    rteb_dataset_name: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure subclasses provide the necessary paths/names
        if not hasattr(self, "rteb_data_path") or not hasattr(
            self, "rteb_dataset_name"
        ):
            raise NotImplementedError(
                "Subclasses of AbsTaskRTEBRetrieval must define class attributes "
                "'rteb_data_path' and 'rteb_dataset_name'"
            )
        if not hasattr(self, "metadata"):
            raise NotImplementedError(
                "Subclasses of AbsTaskRTEBRetrieval must define class attribute 'metadata'"
            )

    def load_data(self, **kwargs: Any) -> None:
        """Data loading is handled by RetrieveDataModule within _evaluate_subset.
        This method can be used for checks or pre-downloads if necessary.
        """
        if self.data_loaded:
            return
        logger.info(
            f"Data for {self.metadata.name} ({self.rteb_dataset_name}) will be loaded "
            f"during evaluation by RTEB's DataModule from path: {self.rteb_data_path}."
        )
        # Optionally check if self.rteb_data_path / self.rteb_dataset_name exists
        # or trigger a download if RTEB doesn't handle it automatically.
        self.data_loaded = True  # Mark as loaded to satisfy MTEB structure

    def _evaluate_subset(
        self,
        model: MTEBEncoder,
        corpus: dict[str, dict[str, str]],  # Not directly used here
        queries: dict[str, str],  # Not directly used here
        relevant_docs: dict[str, dict[str, int]],  # Not directly used here
        hf_subset: HFSubset,  # Not directly used here, relies on self.rteb_dataset_name
        **kwargs: Any,
    ) -> ScoresDict:
        """Evaluate the model using the RTEB evaluation pipeline defined in the base class.
        Uses self.rteb_data_path and self.rteb_dataset_name defined by the subclass.
        """
        logger.info(
            f"Starting RTEB evaluation for {self.metadata.name} using dataset "
            f"{self.rteb_dataset_name} from {self.rteb_data_path}..."
        )

        # 1. Wrap MTEB model
        model_name = getattr(
            model, "model_name", "mteb_wrapped_model"
        )  # Attempt to get name
        rteb_encoder = MTEBToRTEBEncoderWrapper(model, model_name=model_name)

        # 2. Set up RTEB arguments (using defaults, customize as needed)
        args = argparse.Namespace(
            data_path=self.rteb_data_path,  # Uses subclass property
            save_path=kwargs.get(
                "output_folder", f"results/rteb_output/{self.rteb_dataset_name}"
            ),  # Align with MTEB output
            batch_size=kwargs.get("batch_size", 32),
            embd_batch_size=kwargs.get("embd_batch_size", 128),
            num_workers=kwargs.get("num_workers", 4),
            embd_in_memory_threshold=kwargs.get("embd_in_memory_threshold", 100000),
            overwrite=kwargs.get("overwrite_results", False),
            load_embds=False,
            save_embds=False,
            # Add other args required by run_retrieve_task or components if any
        )

        # Ensure save_path exists
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

        # 3. Initialize RTEB components
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            strategy="auto",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        retriever = Retriever(top_k=100)  # Corrected class name
        dataset_meta = DatasetMeta(
            dataset_name=self.rteb_dataset_name
        )  # Uses subclass property

        # 4. Call run_retrieve_task
        rteb_scores = {}
        try:
            # Ensure the encoder has the trainer reference if needed by Lightning hooks
            rteb_encoder._trainer = trainer

            rteb_scores = run_retrieve_task(
                dataset_meta=dataset_meta,
                trainer=trainer,
                encoder=rteb_encoder,
                retriever=retriever,
                args=args,
            )
        except Exception as e:
            logger.error(
                f"Error during RTEB evaluation for {self.metadata.name}: {e}",
                exc_info=True,
            )
        finally:
            # Clean up trainer reference
            rteb_encoder._trainer = None

        if not rteb_scores:
            logger.warning(
                f"RTEB evaluation returned no scores for {self.metadata.name}."
            )
            # Return dummy scores with expected keys for MTEB aggregation
            return {
                "main_score": 0.0,
                self.metadata.main_score: 0.0,
                "hf_subset": hf_subset if self.is_multilingual else "default",
                "languages": self.metadata.eval_langs,
            }

        # 5. Parse results into MTEB ScoresDict format
        mteb_scores = dict(rteb_scores)
        if self.metadata.main_score not in mteb_scores:
            logger.warning(
                f"Main score '{self.metadata.main_score}' not found in RTEB results. "
                f"Available: {list(mteb_scores.keys())}"
            )
            fallback_score = (
                next(iter(mteb_scores.values()), 0.0) if mteb_scores else 0.0
            )
            mteb_scores["main_score"] = fallback_score
        else:
            mteb_scores["main_score"] = mteb_scores[self.metadata.main_score]

        # Remove non-numeric meta keys and ensure float values
        keys_to_remove = ["model_name", "embd_dim", "embd_dtype"]
        final_scores = {}
        for key, value in mteb_scores.items():
            if key not in keys_to_remove:
                try:
                    final_scores[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert score '{key}' value '{value}' to float. Skipping."
                    )

        # Ensure main_score is present even if filtering removed it
        if "main_score" not in final_scores and "main_score" in mteb_scores:
            try:
                final_scores["main_score"] = float(mteb_scores["main_score"])
            except (ValueError, TypeError):
                final_scores["main_score"] = 0.0  # Default if conversion fails

        # Add languages and hf_subset info MTEB expects
        final_scores["hf_subset"] = hf_subset if self.is_multilingual else "default"
        final_scores["languages"] = self.metadata.eval_langs

        logger.info(f"Finished RTEB evaluation for {self.metadata.name}.")
        return final_scores

    # _calculate_metrics_from_split is inherited from AbsTaskRetrieval
    # If descriptive stats are needed, this would need to be implemented,
    # potentially by loading data via RTEB's mechanisms.


# --- End Base Class ---
