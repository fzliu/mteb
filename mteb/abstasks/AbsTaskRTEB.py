from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

# Imports for local file loading - REMOVED
from functools import cache
from pathlib import Path
from time import time
from typing import Any

from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset

from mteb.abstasks.TaskMetadata import HFSubset
from mteb.load_results.task_results import ScoresDict
from mteb.rteb.rteb_task_runner import RTEBTaskRunner

from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

# from mteb.rteb.core.base.dataset import RetrievalDataset # REMOVED
# from mteb.rteb.utils.data import JSONLDataset # REMOVED

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:
    def __init__(
        self,
        hf_repo: str | None = None,
        hf_repo_qrels: str | None = None,
        streaming: bool = False,
        keep_in_memory: bool = False,
        trust_remote_code: bool = False,
        token: str | None = None,
    ):
        self._loaded = False
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.hf_repo = hf_repo
        # By default fetch qrels from same repo not a second repo with "-qrels" like in original
        self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo

        self.streaming = streaming
        self.keep_in_memory = keep_in_memory
        self.trust_remote_code = trust_remote_code

        self.token = token or os.environ["HF_TOKEN"]

    @staticmethod
    def check(fIn: str, ext: str):
        pass  # REMOVED original implementation

    def load(
        self, split="test"
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        if not self._loaded:
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            # logger.info("Doc Example: %s", self.corpus[0]) # Removed as self.corpus is now a Dataset

            logger.info("Loading Queries...")
            self._load_queries()

            self._load_qrels(split)
            self._loaded = True

        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        # Check if qrels is a Dataset before mapping
        if hasattr(self.qrels, "map"):
            self.qrels.map(qrels_dict_init)
        else:
            # If not a Dataset, assume it's already a dict (e.g., from _load_qrels)
            qrels_dict = self.qrels

        # Check if queries is a Dataset before filtering
        if hasattr(self.queries, "filter"):
            self.queries = self.queries.filter(lambda x: x["id"] in qrels_dict)
        # logger.info("Loaded %d %s Queries.", len(self.queries), split.upper()) # Removed as self.queries is now a Dataset
        # logger.info("Query Example: %s", self.queries[0]) # Removed as self.queries is now a Dataset

        return self.corpus, self.queries, qrels_dict  # Return qrels_dict

    def _load_corpus(self):
        corpus_ds = load_dataset(
            self.hf_repo,
            "corpus",
            keep_in_memory=self.keep_in_memory,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )
        corpus_ds = next(iter(corpus_ds.values()))  # get first split
        corpus_ds = corpus_ds.cast_column("id", Value("string"))
        corpus_ds = corpus_ds.remove_columns(
            [col for col in corpus_ds.column_names if col not in ["id", "text"]]
        )
        self.corpus = corpus_ds

    def _load_queries(self):
        queries_ds = load_dataset(
            self.hf_repo,
            "queries",
            keep_in_memory=self.keep_in_memory,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )
        queries_ds = next(iter(queries_ds.values()))  # get first split
        queries_ds = queries_ds.cast_column("id", Value("string"))
        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text"]]
        )
        self.queries = queries_ds

    def _load_qrels(self, split):
        qrels_ds = load_dataset(
            self.hf_repo_qrels,
            keep_in_memory=self.keep_in_memory,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )[split]
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds


class RetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Retrieval"""

    num_samples: int
    num_queries: int
    num_documents: int
    number_of_characters: int

    min_document_length: int
    average_document_length: float
    max_document_length: int
    unique_documents: int

    min_query_length: int
    average_query_length: float
    max_query_length: int
    unique_queries: int

    min_relevant_docs_per_query: int
    average_relevant_docs_per_query: float
    max_relevant_docs_per_query: int
    unique_relevant_docs: int


class AbsTaskRTEB(AbsTask):
    """Abstract class for retrieval experiments."""

    ignore_identical_ids: bool = False
    abstask_prompt = "Retrieve text based on user query."

    def __init__(self, **kwargs):  # Require hf_repo
        self._corpus = None
        self._queries = None
        self._qrels = None

        self.rteb_dataset_name = kwargs.pop("rteb_dataset_name", None)
        # Derive dataset name from task name if not provided
        if self.rteb_dataset_name is None:
            # Remove "RTEB" prefix from task name to get dataset name
            self.rteb_dataset_name = self.metadata.name.replace("RTEB", "")

        self.hf_repo = f"embedding-benchmark/{self.rteb_dataset_name}"
        self._hf_data_loader = HFDataLoader(hf_repo=self.hf_repo)

        super().__init__(**kwargs)

    @property
    @cache
    def corpus(self) -> dict[str, Dataset]:
        self._hf_data_loader.load(split="test")
        return {"test": self._hf_data_loader.corpus}

    @property
    @cache
    def queries(self) -> dict[str, Dataset]:
        self._hf_data_loader.load(split="test")
        return {"test": self._hf_data_loader.queries}

    @property
    @cache
    def relevant_docs(self) -> dict[str, dict[str, dict[str, int]]]:
        # Use the single instance of HFDataLoader
        # HFDataLoader's load method returns corpus, queries, qrels
        # We only need qrels here, and it's already in the desired format
        _, _, qrels = self._hf_data_loader.load(
            split="test"
        )  # Assuming 'test' split for now
        return {"test": qrels}

    def _validate_task_config(self):
        """Validate task-specific configuration."""
        if not self.hf_repo:
            raise ValueError(
                f"HuggingFace repo is required for {self.__class__.__name__}"
            )
        if not self.rteb_dataset_name:
            raise ValueError(
                f"RTEB dataset name is required for {self.__class__.__name__}"
            )

    def load_data(self, **kwargs):
        """Load data from HuggingFace."""
        if self.data_loaded:
            return

        # Validate task configuration
        self._validate_task_config()

        logger.info(
            f"Loading data for {self.metadata.name} ({self.rteb_dataset_name}) from HuggingFace repo: {self.hf_repo}."
        )

        self._hf_data_loader.load()

        # Accessing the properties will trigger the data loading
        _ = self.corpus
        _ = self.queries
        _ = self.relevant_docs

        self.data_loaded = True

    def evaluate(
        self,
        model,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        """Evaluate the model using the RTEB task runner."""
        if not self.data_loaded:
            self.load_data()

        # RTEB tasks handle subsets internally based on dataset name
        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(
                f"Task: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
            )

            scores[hf_subset] = RTEBTaskRunner.run_rteb_evaluation(
                task=self,
                task_metadata=self.metadata,
                rteb_dataset_name=self.rteb_dataset_name,
                model=model,
                hf_subset=hf_subset,
                is_multilingual=self.is_multilingual,
                encode_kwargs=encode_kwargs,
                batch_size=16,
                **kwargs,
            )

        return scores

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ) -> ScoresDict:
        """Evaluate a subset of the dataset.

        This method is required by the base AbsTask class, but the actual evaluation
        logic is delegated to RTEBTaskRunner.run_rteb_evaluation.
        """
        # This method is not used directly in the current implementation
        # as evaluation is delegated to RTEBTaskRunner.
        # However, it must be implemented as it's an abstract method in AbsTask.
        # A minimal implementation that raises NotImplementedError or logs a warning
        # could be used, but keeping the original structure might be safer
        # if there are other parts of the codebase that might still call it.
        # For now, I will restore the original implementation.

        start_time = time()
        results = retriever(corpus, queries)
        end_time = time()
        logger.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

        save_predictions = kwargs.get("save_predictions", False)
        export_errors = kwargs.get("export_errors", False)
        if save_predictions or export_errors:
            output_folder = Path(kwargs.get("output_folder", "results"))
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

        if save_predictions:
            top_k = kwargs.get("top_k", None)
            if top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(
                        sorted(
                            results[qid], key=lambda x: results[qid][x], reverse=True
                        )[:top_k]
                    )
                    results[qid] = {
                        k: v for k, v in results[qid].items() if k in doc_ids
                    }
            qrels_save_path = (
                output_folder / f"{self.metadata.name}_{hf_subset}_predictions.json"
            )

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        ndcg, _map, recall, precision, naucs = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=self.ignore_identical_ids,
        )
        mrr, naucs_mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs.items()
            },
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs_mrr.items()
            },
        }
        self._add_main_score(scores)

        if export_errors:  # TODO
            top_k = kwargs.get("top_k", 1)
            if not save_predictions and top_k == 1:
                for qid in results.keys():
                    doc_scores = results[qid]
                    sorted_docs = sorted(
                        doc_scores.items(), key=lambda x: x[1], reverse=True
                    )[:top_k]
                    results[qid] = dict(sorted_docs)

    def _calculate_metrics_from_split(self, split):
        """Calculate metrics for a given split.

        This method is required by the base AbsTask class, but the actual metric
        calculation is handled within RTEBTaskRunner.run_rteb_evaluation.
        A minimal implementation that raises NotImplementedError or logs a warning
        could be used, but keeping the original structure might be safer
        if there are other parts of the codebase that might still call it.
        For now, I will restore a placeholder implementation.
        """
        # This method is not used directly in the current implementation
        # as metric calculation is delegated to RTEBTaskRunner.
        # However, it must be implemented as it's an abstract method in AbsTask.
        # Returning an empty ScoresDict or raising NotImplementedError are options.
        # For now, returning an empty ScoresDict to satisfy the abstract method requirement.
        logger.warning(
            f"_calculate_metrics_from_split called for split {split}, but metrics are calculated by RTEBTaskRunner."
        )
        return ScoresDict()


def calculate_length(
    corpus: dict[str, dict[str, str]], queries: dict[str, list[str] | str]
) -> RetrievalDescriptiveStatistics:
    """Calculate descriptive statistics for a retrieval dataset."""
    num_queries = sum(len(q) for q in queries.values())
    num_documents = sum(len(c) for c in corpus.values())
    num_samples = num_queries + num_documents

    all_documents = [doc for split in corpus.values() for doc in split.values()]
    all_queries = [query for split in queries.values() for query in split.values()]

    document_lengths = [len(doc) for doc in all_documents]
    query_lengths = [len(query) for query in all_queries]

    min_document_length = min(document_lengths) if document_lengths else 0
    average_document_length = (
        sum(document_lengths) / len(document_lengths) if document_lengths else 0
    )
    max_document_length = max(document_lengths) if document_lengths else 0
    unique_documents = len(set(all_documents))

    min_query_length = min(query_lengths) if query_lengths else 0
    average_query_length = (
        sum(query_lengths) / len(query_lengths) if query_lengths else 0
    )
    max_query_length = max(query_lengths) if query_lengths else 0
    unique_queries = len(set(all_queries))

    # This part requires relevance data, which is not available in this function
    # Setting to default values for now
    min_relevant_docs_per_query = 0
    average_relevant_docs_per_query = 0.0
    max_relevant_docs_per_query = 0
    unique_relevant_docs = 0

    number_of_characters = sum(document_lengths) + sum(query_lengths)

    return RetrievalDescriptiveStatistics(
        num_samples=num_samples,
        num_queries=num_queries,
        num_documents=num_documents,
        number_of_characters=number_of_characters,
        min_document_length=min_document_length,
        average_document_length=average_document_length,
        max_document_length=max_document_length,
        unique_documents=unique_documents,
        min_query_length=min_query_length,
        average_query_length=average_query_length,
        max_query_length=max_query_length,
        unique_queries=unique_queries,
        min_relevant_docs_per_query=min_relevant_docs_per_query,
        average_relevant_docs_per_query=average_relevant_docs_per_query,
        max_relevant_docs_per_query=max_relevant_docs_per_query,
        unique_relevant_docs=unique_relevant_docs,
    )
