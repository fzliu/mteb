# Concrete RTEB task definition for AILAStatutes
from __future__ import annotations

import logging
from typing import Any

# MTEB Imports
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import HFSubset, TaskMetadata
from mteb.encoder_interface import Encoder as MTEBEncoder
from mteb.load_results.task_results import ScoresDict

# RTEB Integration Imports
from mteb.rteb.rteb_task_runner import RTEBTaskRunner  # Import the helper class

logger = logging.getLogger(__name__)


# --- AILAStatutes Specific Task ---
_AILASTATUTES_TASK_NAME = "RTEBAILAStatutes"
_AILASTATUTES_DESCRIPTION = "RTEB evaluation for AILAStatutes dataset."
# Use the user-provided path
_AILASTATUTES_DATA_PATH = "/Users/fodizoltan/Projects/toptal/voyageai/ebr-frank/data"
_AILASTATUTES_DATASET_NAME = "AILAStatutes"
_AILASTATUTES_METADATA = TaskMetadata(
    name=_AILASTATUTES_TASK_NAME,
    description=_AILASTATUTES_DESCRIPTION,
    reference="https://zenodo.org/records/4063986",
    dataset={
        "path": "mteb/AILA_statutes",
        "revision": "ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e",
    },
    type="Retrieval",
    category="s2p",
    eval_splits=["test"],
    eval_langs=["eng-Latn"],  # From text.py groups
    main_score="ndcg_at_10",
    revision="1.0.0",  # Initial revision
    date=None,
    domains=["Legal", "Written"],  # From text.py groups
    task_subtypes=["Article retrieval"],
    license="cc-by-4.0",
    annotations_creators="derived",
    dialect=None,
    text_creation="found",
    bibtex_citation="""@dataset{paheli_bhattacharya_2020_4063986,
  author       = {Paheli Bhattacharya and
                  Kripabandhu Ghosh and
                  Saptarshi Ghosh and
                  Arindam Pal and
                  Parth Mehta and
                  Arnab Bhattacharya and
                  Prasenjit Majumder},
  title        = {AILA 2019 Precedent \& Statute Retrieval Task},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4063986},
  url          = {https://doi.org/10.5281/zenodo.4063986}
}""",
    modalities=["text"],
    hf_subsets_to_langscripts={},
)


class RTEBAILAStatutes(
    AbsTaskRetrieval
):  # Inherit directly from MTEB's AbsTaskRetrieval
    metadata = _AILASTATUTES_METADATA
    # Define RTEB specific paths as class attributes
    rteb_data_path = _AILASTATUTES_DATA_PATH
    rteb_dataset_name = _AILASTATUTES_DATASET_NAME

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


# --- End AILAStatutes Specific Task ---
