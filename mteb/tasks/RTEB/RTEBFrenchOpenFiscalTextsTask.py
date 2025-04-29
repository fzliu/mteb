# Concrete RTEB task definition for FrenchOpenFiscalTexts
from __future__ import annotations

import logging
import os

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFrenchOpenFiscalTexts(AbsTaskRTEB):
    """RTEB task for the FrenchOpenFiscalTexts dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBFrenchOpenFiscalTexts",
        description="RTEB evaluation for FrenchOpenFiscalTexts dataset.",
        reference=None,  # TODO: Add reference URL
        dataset_path="TODO/FrenchOpenFiscalTexts",  # TODO: Verify HF path or if local only
        dataset_revision="main",  # TODO: Verify revision
        eval_langs=["fra-Latn"],  # Assuming French based on name
        main_score="ndcg_at_10",
        revision="1.0.1",  # Increment revision for this refactoring
        date=("YYYY-MM-DD", "YYYY-MM-DD"),  # TODO: Add date range
        domains=["Legal", "Finance"],  # Assuming Legal and Finance based on name
        task_subtypes=[],
        license="unknown",  # TODO: Add license
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""TODO: Add bibtex citation""",
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
            rteb_data_path=rteb_data_path,
            rteb_dataset_name="FrenchOpenFiscalTexts",
            **kwargs,
        )
