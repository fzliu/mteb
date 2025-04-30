from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBMBPP(AbsTaskRTEB):
    """RTEB task for the MBPP dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBMBPP",
        description="RTEB evaluation for MBPP dataset.",
        reference="https://huggingface.co/datasets/Muennighoff/mbpp",
        dataset_path="Muennighoff/mbpp",
        dataset_revision="main",
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        revision="1.0.1",
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        text_creation="found",
        bibtex_citation="""unknown""",
        modalities=["text"],
    )

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="MBPP", **kwargs)
