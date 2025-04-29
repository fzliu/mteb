# Concrete RTEB task definition for TAT_QA
from __future__ import annotations

import logging
import os

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBTAT_QA(AbsTaskRTEB):
    """RTEB task for the TAT_QA dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBTAT_QA",
        description="RTEB evaluation for TAT_QA dataset.",
        reference=None,  # TODO: Add reference URL
        dataset={
            "path": "TODO/TAT_QA",  # TODO: Verify HF path or if local only
            "revision": "main",  # TODO: Verify revision
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],  # Assuming English based on name
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
            rteb_data_path=rteb_data_path, rteb_dataset_name="TAT_QA", **kwargs
        )
