# Concrete RTEB task definition for APPS
from __future__ import annotations

import logging
import os

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB
from mteb.rteb.rteb_utils import create_rteb_task_metadata

logger = logging.getLogger(__name__)


class RTEBAPPS(AbsTaskRTEB):
    """RTEB task for the APPS dataset."""

    metadata = create_rteb_task_metadata(
        task_name="RTEBAPPS",
        description="RTEB evaluation for APPS dataset.",
        reference="https://arxiv.org/abs/2105.09938",
        dataset_path="CoIR-Retrieval/apps",
        dataset_revision="f22508f96b7a36c2415181ed8bb76f76e04ae2d5",
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        revision="1.0.1",  # Increment revision for this refactoring
        date=("2021-05-20", "2021-05-20"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
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
            rteb_data_path=rteb_data_path, rteb_dataset_name="APPS", **kwargs
        )
