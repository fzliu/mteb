# Concrete RTEB task definition for FinanceBench
from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFinanceBench(AbsTaskRTEB):
    """RTEB task for the FinanceBench dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBFinanceBench",
        description="RTEB evaluation for FinanceBench dataset.",
        reference="https://github.com/patronus-ai/financebench",
        dataset_path="PatronusAI/financebench",
        dataset_revision="main",  # Assuming main based on HF page
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        revision="1.0.1",
        date=("2023-11-20", "2023-11-20"),  # Using the date of the arXiv paper
        domains=["Financial"],  # Based on dataset type
        task_subtypes=["Question answering"],
        license="not specified",  # TODO: Verify license
        annotations_creators="human-annotated",
        text_creation="found",
        bibtex_citation="""@misc{islam2023financebench,
      title={FinanceBench: A New Benchmark for Financial Question Answering},
      author={Pranab Islam and Anand Kannappan and Douwe Kiela and Rebecca Qian and Nino Scherrer and Bertie Vidgen},
      year={2023},
      eprint={2311.11944},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",  # Using the bibtex from the GitHub README
        modalities=["text"],
    )

    def __init__(self, **kwargs):
        # Allow configuration via environment variable or default to the original path
        super().__init__(rteb_dataset_name="FinanceBench", **kwargs)
