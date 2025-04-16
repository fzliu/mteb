# Helper class and wrapper for running RTEB evaluation logic (No PyTorch Lightning)
from __future__ import annotations

import argparse
import json  # Needed for saving/loading logic
import logging
import os  # Needed for path checks in replicated logic
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl  # Still needed for LightningModule inheritance
import torch
import torch.distributed  # Needed for replicated logic

# MTEB Imports
from mteb.abstasks.TaskMetadata import HFSubset, TaskMetadata
from mteb.encoder_interface import Encoder as MTEBEncoder
from mteb.encoder_interface import PromptType
from mteb.load_results.task_results import ScoresDict

from .ebr.core.data import RetrieveDataModule  # Need this to load data
from .ebr.core.retriever import Retriever  # Still need the class for similarity_fn
from .ebr.retrieve import run_retrieve_evaluation  # Only need the evaluation part

# RTEB Imports (using relative paths within mteb.rteb)
from .ebr.utils.data import JSONLDataset  # Still needed if we implement save/load
from .ebr.utils.distributed import gather_list

logger = logging.getLogger(__name__)


# --- RTEB Encoder Wrapper (Inheriting LightningModule with __setattr__ override) ---
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


# --- End RTEB Encoder Wrapper ---


# --- RTEB Task Runner Helper ---
class RTEBTaskRunner:
    """Helper class to run RTEB evaluation logic without inheriting MTEB tasks."""

    @staticmethod
    def _encode_data(
        encoder_wrapper: MTEBToRTEBEncoderWrapper,
        dataloader: torch.utils.data.DataLoader,
        task_name: str,  # Add task_name argument
    ) -> dict[str, torch.Tensor]:
        """Manually encodes data using the wrapper."""
        embeddings_dict = {}
        logger.info(
            f"Encoding data for task '{task_name}' using {encoder_wrapper.model_name}..."
        )

        for batch in dataloader:
            # Check for 'text' key instead of 'sentences'
            if not isinstance(batch, dict) or "id" not in batch or "text" not in batch:
                logger.error(
                    f"Unsupported batch type or missing keys ('id', 'text'): {type(batch)} Keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}"
                )
                continue
            ids = batch["id"]
            sentences = batch["text"]  # Use the 'text' key
            if not ids or not sentences:
                continue

            try:
                # Assuming encode returns a tensor of shape [batch_size, emb_dim]
                # Pass task_name as required by some MTEB encoders (like VoyageWrapper)
                # Use the wrapper's encode method, which calls the underlying model's encode
                batch_embeddings = encoder_wrapper.encode(
                    sentences, task_name=task_name, prompt_type=PromptType.passage
                )
                if batch_embeddings.shape[0] != len(ids):
                    logger.error(
                        f"Mismatch between number of IDs ({len(ids)}) and embeddings ({batch_embeddings.shape[0]})"
                    )
                    continue
                for id_val, emb in zip(ids, batch_embeddings):
                    embeddings_dict[id_val] = emb.cpu()  # Store embeddings on CPU
            except Exception as e:
                logger.error(f"Encoding failed for batch: {e}", exc_info=True)
        logger.info(f"Finished encoding. Got {len(embeddings_dict)} embeddings.")
        return embeddings_dict

    @staticmethod
    def _retrieve_scores(
        query_embeddings: dict[str, torch.Tensor],
        corpus_embeddings: dict[str, torch.Tensor],
        retriever: Retriever,  # Use for similarity_fn and topk
    ) -> dict[str, dict[str, float]]:
        """Manually performs retrieval step."""
        all_results = {}
        corpus_ids = list(corpus_embeddings.keys())
        # Stack corpus embeddings into a single tensor for efficient calculation
        # Ensure they are all on the same device (CPU) and float32
        if not corpus_ids:  # Handle empty corpus
            logger.warning("Corpus embeddings are empty, cannot perform retrieval.")
            return {}
        corpus_tensor = torch.stack(list(corpus_embeddings.values())).to(torch.float32)

        logger.info(
            f"Calculating scores for {len(query_embeddings)} queries against {len(corpus_ids)} corpus items..."
        )

        # Determine device for calculation (prefer GPU if available, else CPU)
        device = corpus_tensor.device  # Assume corpus tensor is on target device (CPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):  # Check for MPS
            device = torch.device("mps")

        corpus_tensor = corpus_tensor.to(device)
        logger.info(f"Using device: {device} for score calculation.")

        for qid, query_emb in query_embeddings.items():
            # Ensure query embedding is float32 and move to target device
            query_emb_tensor = (
                query_emb.unsqueeze(0).to(torch.float32).to(device)
            )  # Add batch dim

            # Calculate scores (ensure tensors are on the same device)
            scores = retriever.similarity_fn(query_emb_tensor, corpus_tensor).squeeze(
                0
            )  # Remove batch dim

            # Adjust for distance metrics if needed
            if not retriever.largest:
                scores = scores * -1

            # Get top k
            topk_val = min(retriever.topk, len(corpus_ids))
            if topk_val <= 0:
                continue  # Skip if topk is zero or negative

            # Move scores to CPU before topk if needed, or ensure topk works on device
            top_scores, top_indices = torch.topk(scores.cpu(), topk_val, largest=True)

            # Store results
            query_results = OrderedDict()
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                cid = corpus_ids[idx]
                query_results[cid] = score
            all_results[qid] = query_results

        logger.info("Finished calculating scores.")
        return all_results

    @staticmethod
    def run_rteb_evaluation(
        task_metadata: TaskMetadata,
        rteb_data_path: str,
        rteb_dataset_name: str,
        model: MTEBEncoder,
        hf_subset: HFSubset,
        is_multilingual: bool,
        **kwargs: Any,
    ) -> ScoresDict:
        """Runs the RTEB evaluation pipeline manually without pl.Trainer."""
        logger.info(
            f"Starting RTEB evaluation via Manual Runner: {task_metadata.name} ({rteb_dataset_name})..."
        )

        if hasattr(model, "mteb_model_meta"):
            model_name = model.mteb_model_meta.name
        else:
            model_name = getattr(model, "model_name", "mteb_wrapped_model")
        # Pass save/load flags from kwargs if provided, otherwise default to False
        save_embds_flag = kwargs.get(
            "save_embeddings", False
        )  # Assuming MTEB might pass this
        load_embds_flag = kwargs.get(
            "load_embeddings", False
        )  # Assuming MTEB might pass this

        rteb_encoder = MTEBToRTEBEncoderWrapper(
            model,
            model_name=model_name,
            save_embds=save_embds_flag,
            load_embds=load_embds_flag,
        )

        args = (
            argparse.Namespace(  # Still use args for configuration if needed elsewhere
                data_path=rteb_data_path,
                save_path=kwargs.get(
                    "output_folder", f"results/rteb_output/{rteb_dataset_name}"
                ),
                batch_size=kwargs.get("batch_size", 32),  # Used for dataloader
                embd_batch_size=kwargs.get(
                    "embd_batch_size", 128
                ),  # Not directly used now
                num_workers=kwargs.get(
                    "num_workers", 0
                ),  # Set to 0 for simplicity unless multiprocessing needed
                embd_in_memory_threshold=kwargs.get(
                    "embd_in_memory_threshold", 100000
                ),  # Not directly used now
                overwrite=kwargs.get("overwrite_results", False),
                load_embds=False,  # Simplify: always re-encode for now
                save_embds=False,  # Simplify: don't save intermediate embeddings
            )
        )
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

        # 1. Load Data using RetrieveDataModule
        try:
            dataset_kwargs = {
                "query_instruct": rteb_encoder.query_instruct,
                "corpus_instruct": rteb_encoder.corpus_instruct,
            }
            dm = RetrieveDataModule(
                data_path=args.data_path,
                dataset_name=rteb_dataset_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset_kwargs=dataset_kwargs,
                collator_kwargs={},  # Assuming default collator is fine
            )
            dm.prepare_data()  # Download/prepare data if needed
            logger.info(f"Queries size: {len(dm.dataset.queries)}")
            logger.info(f"Corpus size: {len(dm.dataset.corpus)}")
        except Exception as e:
            logger.error(
                f"Failed to initialize or prepare RetrieveDataModule: {e}",
                exc_info=True,
            )
            return {
                "main_score": 0.0,
                task_metadata.main_score: 0.0,
                "hf_subset": "default",
                "languages": task_metadata.eval_langs,
            }

        # 2. Manually Encode Queries and Corpus
        query_embeddings = RTEBTaskRunner._encode_data(
            rteb_encoder, dm.queries_dataloader(), task_name=task_metadata.name
        )
        corpus_embeddings = RTEBTaskRunner._encode_data(
            rteb_encoder, dm.corpus_dataloader(), task_name=task_metadata.name
        )

        if not query_embeddings or not corpus_embeddings:
            logger.error("Encoding failed, cannot proceed with retrieval.")
            return {
                "main_score": 0.0,
                task_metadata.main_score: 0.0,
                "hf_subset": "default",
                "languages": task_metadata.eval_langs,
            }

        # 3. Manually Perform Retrieval
        retriever_instance = Retriever(
            topk=100
        )  # Instantiate retriever for config/similarity_fn
        predictions = RTEBTaskRunner._retrieve_scores(
            query_embeddings, corpus_embeddings, retriever_instance
        )

        # 4. Run Evaluation
        try:
            # Ensure relevance data is loaded correctly by the datamodule
            relevance_data = dm.dataset.relevance
            if not relevance_data:
                logger.error("Ground truth relevance data not found or empty.")
                raise ValueError("Relevance data is missing.")

            # Filter predictions to only include queries present in relevance data
            filtered_predictions = {
                qid: scores
                for qid, scores in predictions.items()
                if qid in relevance_data
            }
            if len(filtered_predictions) != len(relevance_data):
                logger.warning(
                    f"Number of queries in predictions ({len(filtered_predictions)}) does not match relevance data ({len(relevance_data)}). Evaluating on intersection."
                )
                # Also filter relevance data to match predictions
                filtered_relevance = {
                    qid: scores
                    for qid, scores in relevance_data.items()
                    if qid in filtered_predictions
                }
            else:
                filtered_relevance = relevance_data

            if not filtered_predictions:
                logger.error(
                    "No overlapping queries between predictions and relevance data."
                )
                raise ValueError("No queries to evaluate.")

            rteb_scores = run_retrieve_evaluation(
                filtered_relevance, filtered_predictions
            )
        except Exception as e:
            logger.error(f"Error during score calculation: {e}", exc_info=True)
            rteb_scores = {}  # Ensure it's defined

        # 5. Format and Return Results
        if not rteb_scores:
            logger.warning(
                f"RTEB evaluation returned no scores for {task_metadata.name}."
            )
            return {
                "main_score": 0.0,
                task_metadata.main_score: 0.0,
                "hf_subset": "default",
                "languages": task_metadata.eval_langs,
            }

        mteb_scores = dict(rteb_scores)
        if task_metadata.main_score not in mteb_scores:
            logger.warning(
                f"Main score '{task_metadata.main_score}' not found in RTEB results."
            )
            fallback_score = (
                next(iter(mteb_scores.values()), 0.0) if mteb_scores else 0.0
            )
            mteb_scores["main_score"] = fallback_score
        else:
            mteb_scores["main_score"] = mteb_scores[task_metadata.main_score]

        # Add model info if available from wrapper
        mteb_scores["model_name"] = rteb_encoder.model_name
        if rteb_encoder.embd_dim:
            mteb_scores["embd_dim"] = rteb_encoder.embd_dim
        mteb_scores["embd_dtype"] = rteb_encoder.embd_dtype

        # Remove non-numeric meta keys before returning to MTEB
        keys_to_remove = ["model_name", "embd_dim", "embd_dtype"]
        final_scores = {}
        for key, value in mteb_scores.items():
            if key not in keys_to_remove:
                try:
                    final_scores[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert score '{key}' to float. Skipping."
                    )

        if "main_score" not in final_scores and "main_score" in mteb_scores:
            try:
                final_scores["main_score"] = float(mteb_scores["main_score"])
            except (ValueError, TypeError):
                final_scores["main_score"] = 0.0

        final_scores["hf_subset"] = hf_subset if is_multilingual else "default"
        final_scores["languages"] = task_metadata.eval_langs
        logger.info(f"Finished RTEB evaluation for {task_metadata.name}.")
        return final_scores


# --- End RTEB Task Runner Helper ---
