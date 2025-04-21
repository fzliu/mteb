# Concrete RTEB task definition for APPS
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


# --- APPS Specific Task ---
_APPS_TASK_NAME = "RTEBAPPS"
_APPS_DESCRIPTION = "RTEB evaluation for APPS dataset."
# Use the user-provided path
_APPS_DATA_PATH = "/Users/fodizoltan/Projects/toptal/voyageai/ebr-frank/data"
_APPS_DATASET_NAME = "APPS"
_APPS_METADATA = TaskMetadata(
    name=_APPS_TASK_NAME,
    description=_APPS_DESCRIPTION,
    reference="https://arxiv.org/abs/2105.09938",
    dataset={
        "path": "CoIR-Retrieval/apps",
        "revision": "f22508f96b7a36c2415181ed8bb76f76e04ae2d5",
    },
    type="Retrieval",
    category="s2p",
    eval_splits=["test"],
    eval_langs=["eng-Latn", "python-Code"],
    main_score="ndcg_at_10",
    revision="1.0.0",  # Initial revision
    date=("2021-05-20", "2021-05-20"),
    domains=["Programming", "Written"],
    task_subtypes=["Code retrieval"],
    license="mit",
    annotations_creators="derived",
    dialect=[],
    text_creation="found",
    bibtex_citation="""@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}""",
    modalities=["text"],
    hf_subsets_to_langscripts={},
)


class RTEBAPPS(AbsTaskRetrieval):  # Inherit directly from MTEB's AbsTaskRetrieval
    metadata = _APPS_METADATA
    # Define RTEB specific paths as class attributes
    rteb_data_path = _APPS_DATA_PATH
    rteb_dataset_name = _APPS_DATASET_NAME

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


# --- End APPS Specific Task ---
