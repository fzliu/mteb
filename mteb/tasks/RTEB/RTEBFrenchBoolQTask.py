# Concrete RTEB task definition for FrenchBoolQ
from __future__ import annotations

import logging
import os

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFrenchBoolQ(AbsTaskRTEB):
    """RTEB task for the FrenchBoolQ dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBFrenchBoolQ",
        description="RTEB evaluation for FrenchBoolQ dataset.",
        reference=None,  # TODO: Add reference URL
        dataset={
            "path": "TODO/FrenchBoolQ",  # TODO: Verify HF path or if local only
            "revision": "main",  # TODO: Verify revision
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],  # From text.py groups
        main_score="ndcg_at_10",
        revision="1.0.1",
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
            rteb_data_path=rteb_data_path, rteb_dataset_name="FrenchBoolQ", **kwargs
        )
