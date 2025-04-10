# Content for mteb/tasks/Retrieval/RTEBRetrieval.py (Revision 1)
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np  # Added for type checking
import pytorch_lightning as pl
import torch  # Added for tensor conversion

# MTEB Imports
from mteb.abstasks import AbsTaskRetrieval, TaskMetadata
from mteb.abstasks.TaskMetadata import HFSubset
from mteb.encoder_interface import Encoder as MTEBEncoder  # Renamed to avoid clash
from mteb.load_results.task_results import ScoresDict

# RTEB Imports
from mteb.rteb.ebr.core import Encoder as RtebEncoder
from mteb.rteb.ebr.core.meta import DatasetMeta

# Assuming a default retriever, e.g., DenseRetriever
from mteb.rteb.ebr.core.modules import DenseRetriever
from mteb.rteb.ebr.retrieve import run_retrieve_task

logger = logging.getLogger(__name__)

# --- Metadata (Needs to be updated with actual RTEB dataset info) ---
_TASK_NAME = "RTEBRetrievalExample"  # Needs specific dataset name
_DESCRIPTION = (
    "Integration task for RTEB retrieval using a specific dataset (e.g., NFCorpus)."
)
# Assuming data is stored relative to a base path, needs configuration
_RTEB_DATA_PATH = "data/rteb_datasets"  # Placeholder path
_RTEB_DATASET_NAME = "nfcorpus"  # Example dataset name
_DATASET = {"path": f"{_RTEB_DATA_PATH}/{_RTEB_DATASET_NAME}", "revision": "main"}
_TYPE = "Retrieval"
_CATEGORY = "s2p"
_EVAL_SPLITS = ["test"]
_EVAL_LANGS = ["eng-Latn"]  # Assuming English for NFCorpus
_MAIN_SCORE = "ndcg_at_10"  # Common retrieval metric
_METADATA = TaskMetadata(
    name=_TASK_NAME,
    description=_DESCRIPTION,
    reference="https://github.com/BeIR/beir/wiki",  # Example reference
    dataset=_DATASET,
    type=_TYPE,
    category=_CATEGORY,
    eval_splits=_EVAL_SPLITS,
    eval_langs=_EVAL_LANGS,
    main_score=_MAIN_SCORE,
    revision="1.0.0",  # Placeholder revision
    date=("2024-01-01", "2024-01-01"),  # Placeholder date
    form=["written"],
    domains=["Web", "Medical"],  # Example domains for NFCorpus
    task_subtypes=[],
    license="apache-2.0",  # Example license
    socioeconomic_status="mixed",
    annotations_creators="derived",
    dialect=[],
    text_creation="found",
    bibtex_citation="""@misc{placeholder_rteb, title={RTEB Placeholder}}""",  # Needs real citation
    n_samples={"test": 3633},  # Example count for NFCorpus test
    avg_character_length={"test": 1000},  # Placeholder char length
    modalities=["text"],
    hf_subsets_to_langscripts={},
)
# --- End Metadata ---


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
            self.embd_dim = embeddings.shape[1]

        # Ensure output is torch.Tensor
        if isinstance(embeddings, np.ndarray):
            return torch.from_numpy(embeddings)
        elif isinstance(embeddings, torch.Tensor):
            return embeddings
        elif isinstance(
            embeddings, list
        ):  # Handle list of tensors/arrays if model returns that
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
            return self.encode(batch["sentences"])
        elif isinstance(batch, list):  # Assuming batch is just a list of sentences
            return self.encode(batch)
        else:
            raise TypeError(f"Unsupported batch type in predict_step: {type(batch)}")

    # Potentially add other methods required by RtebEncoder or pl.LightningModule if any


# --- End RTEB Encoder Wrapper ---


class RTEBRetrieval(AbsTaskRetrieval):
    metadata = _METADATA

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store RTEB specific paths/configs if needed
        self.rteb_data_path = kwargs.get("rteb_data_path", _RTEB_DATA_PATH)
        self.rteb_dataset_name = kwargs.get("rteb_dataset_name", _RTEB_DATASET_NAME)

    def load_data(self, **kwargs: Any) -> None:
        """Data loading is handled by RetrieveDataModule within _evaluate_subset.
        This method can be used for checks or pre-downloads if necessary.
        """
        if self.data_loaded:
            return
        logger.info(
            f"Data for {self.metadata.name} will be loaded during evaluation by RTEB's DataModule."
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
        hf_subset: HFSubset,  # Map this to RTEB dataset if needed, currently using self.rteb_dataset_name
        **kwargs: Any,
    ) -> ScoresDict:
        """Evaluate the model using the RTEB evaluation pipeline."""
        logger.info(f"Starting RTEB evaluation for {self.metadata.name}...")

        # 1. Wrap MTEB model
        # TODO: Pass model name properly if available from MTEB context
        model_name = getattr(
            model, "model_name", "mteb_wrapped_model"
        )  # Attempt to get name
        rteb_encoder = MTEBToRTEBEncoderWrapper(model, model_name=model_name)

        # 2. Set up RTEB arguments (using defaults, customize as needed)
        # Using a simple Namespace object for compatibility with run_retrieve_task
        args = argparse.Namespace(
            data_path=self.rteb_data_path,
            save_path=kwargs.get(
                "output_folder", "results/rteb_output"
            ),  # Align with MTEB output if possible
            batch_size=kwargs.get("batch_size", 32),  # Get from MTEB kwargs if passed
            embd_batch_size=kwargs.get("embd_batch_size", 128),
            num_workers=kwargs.get("num_workers", 4),
            embd_in_memory_threshold=kwargs.get("embd_in_memory_threshold", 100000),
            overwrite=kwargs.get("overwrite_results", False),  # Get from MTEB kwargs
            load_embds=False,  # Default to re-computing embeddings
            save_embds=False,  # Default to not saving embeddings
            # Add other args required by run_retrieve_task or components if any
        )

        # Ensure save_path exists
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

        # 3. Initialize RTEB components
        # Trainer (using minimal config)
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",  # Use "auto" or specify e.g., [0] for GPU 0
            strategy="auto",
            logger=False,  # Disable PL logging unless needed
            enable_checkpointing=False,
            enable_progress_bar=True,  # Show progress bars
            enable_model_summary=False,
        )

        # Retriever (using DenseRetriever as example)
        # TODO: Configure retriever properly (e.g., top_k)
        retriever = DenseRetriever(top_k=100)  # Example top_k

        # Dataset Meta
        dataset_meta = DatasetMeta(
            dataset_name=self.rteb_dataset_name
        )  # Use the configured name

        # 4. Call run_retrieve_task
        # Note: run_retrieve_task handles DataModule setup internally based on args
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
        except NotImplementedError as e:
            logger.error(f"Missing implementation in RTEB wrapper: {e}")
            # Return dummy scores on error during development
            rteb_scores = {}
        except Exception as e:
            logger.error(
                f"Error during RTEB evaluation for {self.metadata.name}: {e}",
                exc_info=True,
            )  # Log traceback
            # Optionally re-raise or return dummy scores
            rteb_scores = {}  # Return empty scores on failure
        finally:
            # Clean up trainer reference
            rteb_encoder._trainer = None

        if not rteb_scores:
            logger.warning(
                f"RTEB evaluation returned no scores for {self.metadata.name}."
            )
            return {
                "main_score": 0.0,
                self.metadata.main_score: 0.0,
            }  # Return dummy scores

        # 5. Parse results into MTEB ScoresDict format
        # run_retrieve_evaluation already calculates ndcg@k, map@k etc.
        # We just need to ensure the keys match MTEB expectations if needed,
        # and add the 'main_score'.
        mteb_scores = dict(rteb_scores)  # Copy the scores
        if self.metadata.main_score not in mteb_scores:
            logger.warning(
                f"Main score '{self.metadata.main_score}' not found in RTEB results. Available: {list(mteb_scores.keys())}"
            )
            # Assign a default or fallback score if main score is missing
            fallback_score = (
                next(iter(mteb_scores.values()), 0.0) if mteb_scores else 0.0
            )
            mteb_scores["main_score"] = fallback_score
            # Do not add the specific key if missing, main_score is the generic one
            # mteb_scores[self.metadata.main_score] = fallback_score
        else:
            mteb_scores["main_score"] = mteb_scores[self.metadata.main_score]

        # Remove non-numeric meta keys added by RTEB if necessary
        keys_to_remove = ["model_name", "embd_dim", "embd_dtype"]
        final_scores = {}
        for key, value in mteb_scores.items():
            if key not in keys_to_remove:
                # Ensure value is json-serializable (float)
                try:
                    final_scores[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert score '{key}' value '{value}' to float. Skipping."
                    )

        logger.info(f"Finished RTEB evaluation for {self.metadata.name}.")
        # Ensure main_score is present even if filtering removed it
        if "main_score" not in final_scores and "main_score" in mteb_scores:
            try:
                final_scores["main_score"] = float(mteb_scores["main_score"])
            except (ValueError, TypeError):
                final_scores["main_score"] = 0.0  # Default if conversion fails

        # Add languages and hf_subset info MTEB expects
        final_scores["hf_subset"] = hf_subset if self.is_multilingual else "default"
        final_scores["languages"] = (
            self.metadata.eval_langs
        )  # Assuming single lang for now

        return final_scores

    # TODO: Implement _calculate_metrics_from_split if needed for descriptive stats
    # This would require loading data similar to how AbsTaskRetrieval does it,
    # potentially duplicating effort or needing access to RTEB's loaded data.
    # For now, inheriting the base implementation which raises NotImplementedError is fine.
