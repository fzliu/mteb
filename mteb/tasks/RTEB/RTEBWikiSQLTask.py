from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBWikiSQL(AbsTaskRTEB):
    """RTEB task for the WikiSQL dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBWikiSQL",
        description="RTEB evaluation for WikiSQL dataset.",
        reference="https://huggingface.co/datasets/Salesforce/wikisql",
        dataset_path="Salesforce/wikisql",
        dataset_revision="main",
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        revision="1.0.1",
        domains=["Programming"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        text_creation="found",
        bibtex_citation="""unknown""",
        modalities=["text"],
    )

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="WikiSQL", **kwargs)
