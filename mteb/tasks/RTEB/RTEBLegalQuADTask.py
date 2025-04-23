# Concrete RTEB task definition for LegalQuAD
from __future__ import annotations

import logging
import os

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB
from mteb.rteb.rteb_utils import create_rteb_task_metadata

logger = logging.getLogger(__name__)


class RTEBLegalQuAD(AbsTaskRTEB):
    """RTEB task for the LegalQuAD dataset."""

    metadata = create_rteb_task_metadata(
        task_name="RTEBLegalQuAD",
        description="RTEB evaluation for LegalQuAD dataset.",
        reference="https://github.com/elenanereiss/LegalQuAD",
        dataset_path="mteb/LegalQuAD",
        dataset_revision="dd73c838031a4914a7a1a16d785b8cec617aaaa4",
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        revision="1.0.5",  # Increment revision for this refactoring
        date=("2021-11-01", "2021-11-01"),
        domains=["Legal"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        text_creation="found",
        bibtex_citation="""@inproceedings{reiss-etal-2021-legalquad, ... }""",  # Truncated
        modalities=["text"],
    )

    def __init__(self, **kwargs):
        # Allow configuration via environment variable or default to the original path
        rteb_data_path = kwargs.pop(
            "rteb_data_path",
            os.environ.get(
                "RTEB_DATA_PATH",
                "/Users/fodizoltan/Projects/toptal/voyageai/ebr-frank/data",
            ),
        )
        super().__init__(
            rteb_data_path=rteb_data_path, rteb_dataset_name="LegalQuAD", **kwargs
        )
