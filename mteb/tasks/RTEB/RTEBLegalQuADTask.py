from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBLegalQuAD(AbsTaskRTEB):
    """RTEB task for the LegalQuAD dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBLegalQuAD",
        description="RTEB evaluation for LegalQuAD dataset.",
        reference="https://github.com/elenanereiss/LegalQuAD",
        dataset_path="mteb/LegalQuAD",
        dataset_revision="dd73c838031a4914a7a1a16d785b8cec617aaaa4",
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        revision="1.0.0",
        date=("2021-11-01", "2021-11-01"),
        domains=["Legal"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        text_creation="found",
        bibtex_citation="""@inproceedings{reiss-etal-2021-legalquad, ... }""",
        modalities=["text"],
    )

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="LegalQuAD", **kwargs)
