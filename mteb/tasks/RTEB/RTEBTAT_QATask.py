from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBTAT_QA(AbsTaskRTEB):
    """RTEB task for the TAT_QA dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBTAT_QA",
        description="RTEB evaluation for TAT_QA dataset.",
        reference="https://huggingface.co/datasets/next-tat/TAT-QA",
        dataset_path="next-tat/TAT-QA",
        dataset_revision="main",
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        revision="1.0.1",
        domains=["Financial"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        text_creation="found",
        bibtex_citation="""unknown""",
        modalities=["text"],
    )

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="TAT_QA", **kwargs)
