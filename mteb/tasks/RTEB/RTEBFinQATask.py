from __future__ import annotations

import logging

from mteb.abstasks.AbsTaskRTEB import AbsTaskRTEB

logger = logging.getLogger(__name__)


class RTEBFinQA(AbsTaskRTEB):
    """RTEB task for the FinQA dataset."""

    metadata = AbsTaskRTEB.create_rteb_task_metadata(
        task_name="RTEBFinQA",
        description="RTEB evaluation for FinQA dataset.",
        reference="https://finqasite.github.io/",
        dataset_path="ibm-research/finqa",
        dataset_revision="main",
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        revision="1.0.1",
        date=("2021-09-01", "2021-09-01"),
        domains=["Financial"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="expert-annotated",
        text_creation="found",
        bibtex_citation="""@article{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Wenhu and Chen, Zhiyu and Wang, Chuhan and Zhang, Xinyi and Zhang, Yuchi and Smrz, Pavel and Yu, Xiangyu and Fung, Pascale},
  journal={arXiv preprint arXiv:2109.00122},
  year={2021}
}""",
        modalities=["text"],
    )

    def __init__(self, **kwargs):
        super().__init__(rteb_dataset_name="FinQA", **kwargs)
