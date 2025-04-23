from __future__ import annotations

import logging
from typing import Any

from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


def create_rteb_task_metadata(
    task_name: str,
    dataset_name: str | None = None,
    description: str | None = None,
    reference: str | None = None,
    dataset_path: str | None = None,
    dataset_revision: str | None = None,
    eval_langs: list[str] | None = None,
    main_score: str = "ndcg_at_10",
    domains: list[str] | None = None,
    revision: str = "1.0.0",
    date: tuple[str, str] | None = None,
    license: str | None = None,
    annotations_creators: str | None = None,
    text_creation: str | None = None,
    task_subtypes: list[str] | None = None,
    dialect: list[str] | None = None,
    bibtex_citation: str | None = None,
    modalities: list[str] | None = None,
    hf_subsets_to_langscripts: dict[str, list[str]] | None = None,
    **kwargs: Any,
) -> TaskMetadata:
    """Factory function to create TaskMetadata for RTEB tasks with sensible defaults.

    This function simplifies the creation of TaskMetadata objects for RTEB tasks
    by providing sensible defaults and deriving values where possible.

    Args:
        task_name: Name of the task (e.g., "RTEBLegalQuAD")
        dataset_name: Name of the dataset. If None, derived from task_name by removing "RTEB" prefix
        description: Task description. If None, generated from dataset_name
        reference: Reference URL for the dataset
        dataset_path: HuggingFace dataset path. If None, defaults to "mteb/{dataset_name}"
        dataset_revision: HuggingFace dataset revision
        eval_langs: List of evaluation languages. Defaults to ["eng-Latn"]
        main_score: Main evaluation metric. Defaults to "ndcg_at_10"
        domains: List of domains the dataset belongs to
        revision: Task revision string
        date: Tuple of (start_date, end_date) for the dataset
        license: Dataset license
        annotations_creators: How annotations were created
        text_creation: How text was created
        task_subtypes: List of task subtypes
        dialect: List of dialects
        bibtex_citation: BibTeX citation for the dataset
        modalities: List of modalities
        hf_subsets_to_langscripts: Mapping of HF subsets to language scripts
        **kwargs: Additional arguments to pass to TaskMetadata

    Returns:
        TaskMetadata object configured for the RTEB task
    """
    # Derive dataset name from task name if not provided
    if dataset_name is None:
        dataset_name = task_name.replace("RTEB", "")

    # Generate description if not provided
    if description is None:
        description = f"RTEB evaluation for {dataset_name} dataset."

    # Set default dataset path if not provided
    if dataset_path is None:
        dataset_path = f"mteb/{dataset_name}"

    # Set default date if not provided
    if date is None:
        date = ("2021-01-01", "2021-01-01")

    # Set default eval_langs if not provided
    if eval_langs is None:
        eval_langs = ["eng-Latn"]

    # Set default domains if not provided
    if domains is None:
        domains = []

    # Set default task_subtypes if not provided
    if task_subtypes is None:
        task_subtypes = []

    # Set default dialect if not provided
    if dialect is None:
        dialect = []

    # Set default modalities if not provided
    if modalities is None:
        modalities = ["text"]

    # Set default hf_subsets_to_langscripts if not provided
    if hf_subsets_to_langscripts is None:
        hf_subsets_to_langscripts = {}

    # Create dataset dictionary
    dataset_dict = {"path": dataset_path}
    if dataset_revision:
        dataset_dict["revision"] = dataset_revision

    # Create and return TaskMetadata
    return TaskMetadata(
        name=task_name,
        description=description,
        reference=reference,
        dataset=dataset_dict,
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=eval_langs,
        main_score=main_score,
        revision=revision,
        date=date,
        domains=domains,
        license=license,
        annotations_creators=annotations_creators,
        text_creation=text_creation,
        task_subtypes=task_subtypes,
        dialect=dialect,
        bibtex_citation=bibtex_citation,
        modalities=modalities,
        hf_subsets_to_langscripts=hf_subsets_to_langscripts,
        **kwargs,
    )
