# Concrete RTEB task definitions
from __future__ import annotations

import logging

# MTEB Imports
from mteb.abstasks.TaskMetadata import TaskMetadata  # Keep this for metadata definition

# Local RTEB Integration Imports
from mteb.rteb.rteb_base_task import (
    AbsTaskRTEBRetrieval,
)  # Import base class from its new location

logger = logging.getLogger(__name__)


# --- LegalQuAD Specific Task ---
_LEGALQUAD_TASK_NAME = "RTEBLegalQuAD"
_LEGALQUAD_DESCRIPTION = "RTEB evaluation for LegalQuAD dataset."
# Use the user-provided path
_LEGALQUAD_DATA_PATH = "/Users/fodizoltan/Projects/toptal/voyageai/ebr-frank/data"
_LEGALQUAD_DATASET_NAME = "LegalQuAD"
_LEGALQUAD_METADATA = TaskMetadata(
    name=_LEGALQUAD_TASK_NAME,
    description=_LEGALQUAD_DESCRIPTION,
    reference="https://github.com/elenanereiss/LegalQuAD",
    # MTEB reference path is informational here as RTEB loads data differently
    dataset={
        "path": "mteb/LegalQuAD",
        "revision": "dd73c838031a4914a7a1a16d785b8cec617aaaa4",
    },
    type="Retrieval",
    category="s2p",
    eval_splits=["test"],
    eval_langs=["deu-Latn"],
    main_score="ndcg_at_10",
    revision="1.0.2",  # Increment revision for this refactoring
    date=("2021-11-01", "2021-11-01"),
    form=["written"],
    domains=["Legal"],
    task_subtypes=[],
    license="cc-by-nc-sa-4.0",
    socioeconomic_status="high",
    annotations_creators="derived",
    dialect=[],
    text_creation="found",
    bibtex_citation="""@inproceedings{reiss-etal-2021-legalquad,
    title = "{L}egal{Q}u{AD}: A Question Answering Dataset for the {G}erman Legal Domain",
    author = "Rei{\ss}, Elena  and
      Grabow, Christoph  and
      Schumann, Anne-Kathrin",
    editor = "Ntoutsi, Eirini  and
      Fafalios, Pavlos  and
      Huber, Brigitte  and
      Lange, Dimitar  and
      Teije, Annette ten  and
      Vahdati, Sahar  and
      Vargas-Vera, Maria  and
      Lehmann, Jens",
    booktitle = "Joint Proceedings of the Semantics and Knowledge Graphs track at the {ESWC} 2021",
    month = jun,
    year = "2021",
    address = "Hersonissos, Greece",
    publisher = "{CEUR} Workshop Proceedings",
    url = "https://ceur-ws.org/Vol-2934/paper1.pdf",
    volume = "2934",
    pages = "1--15",
}""",
    n_samples={"test": 1000},  # Adjust if your test set size differs
    avg_character_length={"test": 1198.6},
    modalities=["text"],
    hf_subsets_to_langscripts={},
)


class RTEBLegalQuAD(AbsTaskRTEBRetrieval):
    metadata = _LEGALQUAD_METADATA
    rteb_data_path = _LEGALQUAD_DATA_PATH
    rteb_dataset_name = _LEGALQUAD_DATASET_NAME

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# --- End LegalQuAD Specific Task ---


# --- Add other dataset subclasses similarly below ---
# e.g.
# class RTEBNFCorpus(AbsTaskRTEBRetrieval):
#     metadata = ...
#     rteb_data_path = ...
#     rteb_dataset_name = "nfcorpus"
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
