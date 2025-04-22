# Concrete RTEB task definition for MBPP
from __future__ import annotations

import logging
from typing import Any

# MTEB Imports
from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB
from mteb.abstasks.TaskMetadata import HFSubset, TaskMetadata
from mteb.encoder_interface import Encoder as MTEBEncoder
from mteb.load_results.task_results import ScoresDict

# RTEB Integration Imports
from mteb.rteb.rteb_task_runner import RTEBTaskRunner  # Import the helper class

logger = logging.getLogger(__name__)


# --- MBPP Specific Task ---
_MBPP_TASK_NAME = "RTEBMBPP"
_MBPP_DESCRIPTION = "RTEB evaluation for MBPP dataset."
# Use the user-provided path
_MBPP_DATA_PATH = "/Users/fodizoltan/Projects/toptal/voyageai/ebr-frank/data"
_MBPP_DATASET_NAME = "MBPP"
_MBPP_METADATA = TaskMetadata(
    name=_MBPP_TASK_NAME,
    description=_MBPP_DESCRIPTION,
    reference=None,  # TODO: Add reference URL
    dataset={
        "path": "TODO/MBPP",  # TODO: Verify HF path or if local only
        "revision": "main",  # TODO: Verify revision
    },
    type="Retrieval",
    category="s2p",
    eval_splits=["test"],
    eval_langs=["eng-Latn"],  # Assuming English based on name
    main_score="ndcg_at_10",
    revision="1.0.0",  # Initial revision
    date=("YYYY-MM-DD", "YYYY-MM-DD"),  # TODO: Add date range
    domains=["Code"],  # From text.py groups
    task_subtypes=[],
    license="unknown",  # TODO: Add license
    annotations_creators="derived",  # Assuming similar to example
    dialect=[],
    text_creation="found",  # Assuming similar to example
    bibtex_citation="""TODO: Add bibtex citation""",
    modalities=["text"],
    hf_subsets_to_langscripts={},
)


class RTEBMBPP(AbsTaskRTEB):  # Inherit directly from MTEB's AbsTaskRTEB
    metadata = _MBPP_METADATA
    # Define RTEB specific paths as class attributes
    rteb_data_path = _MBPP_DATA_PATH
    rteb_dataset_name = _MBPP_DATASET_NAME

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs: Any) -> None:
        """Data loading is handled by the RTEB runner.
        Mark data as loaded to satisfy MTEB's checks.
        """
        if self.data_loaded:
            return
        logger.info(
            f"Data for {self.metadata.name} ({self.rteb_dataset_name}) will be loaded "
            f"during evaluation by RTEB's runner from path: {self.rteb_data_path}."
        )
        self.data_loaded = True

    def evaluate(
        self,
        model: MTEBEncoder,
        split: str = "test",
        *,
        encode_kwargs: dict[
            str, Any
        ] = {},  # Keep encode_kwargs for potential future use
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        """Override the base evaluate method to call the RTEB runner."""
        if not self.data_loaded:
            self.load_data()

        # RTEB tasks handle subsets internally based on dataset name,
        # so we evaluate only the 'default' subset here which triggers the runner.
        hf_subset = "default"
        logger.info(
            f"Task: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
        )

        # Pass necessary info to the static runner method
        # Note: corpus, queries, relevant_docs from the base class evaluate signature are ignored here.
        scores = {
            hf_subset: RTEBTaskRunner.run_rteb_evaluation(
                task_metadata=self.metadata,
                rteb_data_path=self.rteb_data_path,
                rteb_dataset_name=self.rteb_dataset_name,
                model=model,
                hf_subset=hf_subset,
                is_multilingual=self.is_multilingual,
                **kwargs,  # Pass other MTEB kwargs like output_folder
            )
        }
        return scores


# --- End MBPP Specific Task ---
