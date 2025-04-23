# Concrete RTEB task definition for AILAStatutes
from __future__ import annotations

import logging
import os

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB
from mteb.rteb.rteb_utils import create_rteb_task_metadata

logger = logging.getLogger(__name__)


class RTEBAILAStatutes(AbsTaskRTEB):
    """RTEB task for the AILAStatutes dataset."""

    metadata = create_rteb_task_metadata(
        task_name="RTEBAILAStatutes",
        description="RTEB evaluation for AILAStatutes dataset.",
        reference="https://zenodo.org/records/4063986",
        dataset_path="mteb/AILA_statutes",
        dataset_revision="ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e",
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        revision="1.0.1",  # Increment revision for this refactoring
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        text_creation="found",
        bibtex_citation="""@dataset{paheli_bhattacharya_2020_4063986,
  author       = {Paheli Bhattacharya and
                  Kripabandhu Ghosh and
                  Saptarshi Ghosh and
                  Arindam Pal and
                  Parth Mehta and
                  Arnab Bhattacharya and
                  Prasenjit Majumder},
  title        = {AILA 2019 Precedent \\& Statute Retrieval Task},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4063986},
  url          = {https://doi.org/10.5281/zenodo.4063986}
}""",
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
            rteb_data_path=rteb_data_path, rteb_dataset_name="AILAStatutes", **kwargs
        )
