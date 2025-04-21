from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.utils.data

from mteb.abstasks.TaskMetadata import HFSubset, TaskMetadata
from mteb.encoder_interface import Encoder as MTEBEncoder
from mteb.encoder_interface import PromptType
from mteb.load_results.task_results import ScoresDict
from mteb.rteb.core.data import RetrieveDataModule
from mteb.rteb.core.retriever import Retriever
from mteb.rteb.retrieve import (
    CORPUS_EMBD_FILENAME,
    QUERIES_EMBD_FILENAME,
    RETRIEVE_EVAL_FILENAME,
    RETRIEVE_PRED_FILENAME,
    run_retrieve_evaluation,
)
from mteb.rteb.rteb_encoder_wrapper import (
    MTEBToRTEBEncoderWrapper,
)  # Import the new wrapper file

logger = logging.getLogger(__name__)


class RTEBTaskRunner:
    """Helper class to run RTEB evaluation logic without inheriting MTEB tasks."""

    @staticmethod
    def _encode_data(
        encoder_wrapper: MTEBToRTEBEncoderWrapper,
        dataloader: torch.utils.data.DataLoader,
        task_name: str,
        prompt_type: PromptType,
    ) -> dict[str, torch.Tensor]:
        """Manually encodes data using the wrapper."""
        embeddings_dict = {}
        logger.info(
            f"Encoding data for task '{task_name}' using {encoder_wrapper.model_name}..."
        )

        for batch in dataloader:
            if not isinstance(batch, dict) or "id" not in batch or "text" not in batch:
                logger.error(
                    f"Unsupported batch type or missing keys ('id', 'text'): {type(batch)} Keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}"
                )
                continue
            ids = batch["id"]
            sentences = batch["text"]
            if not ids or not sentences:
                continue

            try:
                batch_embeddings = encoder_wrapper.encode(
                    sentences, task_name=task_name, prompt_type=prompt_type
                )
                if batch_embeddings.shape[0] != len(ids):
                    logger.error(
                        f"Mismatch between number of IDs ({len(ids)}) and embeddings ({batch_embeddings.shape[0]})"
                    )
                    continue
                for id_val, emb in zip(ids, batch_embeddings):
                    embeddings_dict[id_val] = emb.cpu()
            except Exception as e:
                logger.error(f"Encoding failed for batch: {e}", exc_info=True)
        logger.info(f"Finished encoding. Got {len(embeddings_dict)} embeddings.")
        return embeddings_dict

    @staticmethod
    def _retrieve_scores(
        query_embeddings: dict[str, torch.Tensor],
        corpus_embeddings: dict[str, torch.Tensor],
        retriever: Retriever,
    ) -> dict[str, dict[str, float]]:
        """Manually performs retrieval step."""
        all_results = {}
        corpus_ids = list(corpus_embeddings.keys())
        if not corpus_ids:
            logger.warning("Corpus embeddings are empty, cannot perform retrieval.")
            return {}
        corpus_tensor = torch.stack(list(corpus_embeddings.values())).to(torch.float32)

        logger.info(
            f"Calculating scores for {len(query_embeddings)} queries against {len(corpus_ids)} corpus items..."
        )

        device = corpus_tensor.device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")

        corpus_tensor = corpus_tensor.to(device)
        logger.info(f"Using device: {device} for score calculation.")

        for qid, query_emb in query_embeddings.items():
            query_emb_tensor = query_emb.unsqueeze(0).to(torch.float32).to(device)

            scores = retriever.similarity_fn(query_emb_tensor, corpus_tensor).squeeze(0)

            if not retriever.largest:
                scores = scores * -1

            topk_val = min(retriever.topk, len(corpus_ids))
            if topk_val <= 0:
                continue

            top_scores, top_indices = torch.topk(scores.cpu(), topk_val, largest=True)

            query_results = OrderedDict()
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                cid = corpus_ids[idx]
                query_results[cid] = score
            all_results[qid] = query_results

        logger.info("Finished calculating scores.")
        return all_results

    @staticmethod
    def run_rteb_evaluation(
        task_metadata: TaskMetadata,
        rteb_data_path: str,
        rteb_dataset_name: str,
        model: MTEBEncoder,
        hf_subset: HFSubset,
        is_multilingual: bool,
        **kwargs: Any,
    ) -> ScoresDict:
        """Runs the RTEB evaluation pipeline with pl.Trainer."""
        logger.info(
            f"Starting RTEB evaluation via Manual Runner: {task_metadata.name} ({rteb_dataset_name})..."
        )

        if hasattr(model, "mteb_model_meta"):
            model_name = model.mteb_model_meta.name
        else:
            model_name = getattr(model, "model_name", "mteb_wrapped_model")

        # Configure Trainer
        trainer_kwargs = {
            "accelerator": kwargs.get("accelerator", "auto"),
            "devices": kwargs.get("devices", "auto"),
            "num_nodes": kwargs.get("num_nodes", 1),
            "strategy": kwargs.get("strategy", "auto"),
            "precision": kwargs.get("precision", "32-true"),
            "logger": False,  # Disable default logger
            "enable_checkpointing": False,
            "enable_progress_bar": True,
        }
        trainer = pl.Trainer(**trainer_kwargs)

        save_embds_flag = kwargs.get("save_embeddings", False)
        load_embds_flag = kwargs.get("load_embeddings", False)

        rteb_encoder = MTEBToRTEBEncoderWrapper(
            model,
            task_name=task_metadata.name,
            model_name=model_name,
            save_embds=save_embds_flag,
            load_embds=load_embds_flag,
        )
        rteb_encoder._trainer = trainer

        args = argparse.Namespace(
            data_path=rteb_data_path,
            save_path=kwargs.get(
                "output_folder", f"results/rteb_output/{rteb_dataset_name}"
            ),
            batch_size=kwargs.get("batch_size", 16),
            embd_batch_size=kwargs.get("embd_batch_size", 128),
            num_workers=kwargs.get("num_workers", 0),
            embd_in_memory_threshold=kwargs.get("embd_in_memory_threshold", 100000),
            overwrite=kwargs.get("overwrite_results", False),
            load_embds=load_embds_flag,  # Use the flag from kwargs
            save_embds=save_embds_flag,  # Use the flag from kwargs
        )
        task_save_path = Path(args.save_path) / model_name
        task_save_path.mkdir(parents=True, exist_ok=True)

        # Check if results already exist
        eval_file = task_save_path / RETRIEVE_EVAL_FILENAME  # Use consistent filename
        if not args.overwrite and eval_file.exists():
            if trainer.is_global_zero:
                logger.info(
                    f"Results already exist for {task_metadata.name} at {eval_file}. Skipping."
                )
                with open(str(eval_file)) as f:
                    scores = json.load(f)
                return scores
            else:
                # Non-global zero ranks should wait for global zero to finish
                trainer.strategy.barrier()
                with open(str(eval_file)) as f:
                    scores = json.load(f)
                return scores

        # 1. Load Data using RetrieveDataModule
        try:
            dataset_kwargs = {
                "query_instruct": rteb_encoder.query_instruct,
                "corpus_instruct": rteb_encoder.corpus_instruct,
            }
            dm = RetrieveDataModule(
                data_path=args.data_path,
                dataset_name=rteb_dataset_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset_kwargs=dataset_kwargs,
                collator_kwargs={},
            )
            if trainer.is_global_zero:
                dm.prepare_data()
                logger.info(f"Queries size: {len(dm.dataset.queries)}")
                logger.info(f"Corpus size: {len(dm.dataset.corpus)}")

            trainer.strategy.barrier()  # Ensure data is prepared on all ranks

            if (
                len(dm.dataset.queries) < trainer.num_devices
                or len(dm.dataset.corpus) < trainer.num_devices
            ):
                logger.warning("Skipping the task due to too few queries / documents.")
                return {}

            if len(dm.dataset.queries) >= 1e6:
                logger.warning("Skipping the task due to too many queries.")
                return {}
        except Exception as e:
            logger.error(
                f"Failed to initialize or prepare RetrieveDataModule: {e}",
                exc_info=True,
            )
            return {
                "main_score": 0.0,
                task_metadata.main_score: 0.0,
                "hf_subset": "default",
                "languages": task_metadata.eval_langs,
            }

        # 2. Encode Queries and Corpus using pl.Trainer
        queries_embds_file = (
            task_save_path / QUERIES_EMBD_FILENAME
        )  # Use consistent filename
        corpus_embds_file = (
            task_save_path / CORPUS_EMBD_FILENAME
        )  # Use consistent filename

        # Encode Queries
        logger.info("Encoding queries")
        rteb_encoder.is_query = True
        rteb_encoder.in_memory = len(dm.dataset.queries) < args.embd_in_memory_threshold
        rteb_encoder.save_file = os.path.join(task_save_path, QUERIES_EMBD_FILENAME)
        if args.load_embds and rteb_encoder.embd_files_exist(trainer.num_devices):
            queries_embds_files = rteb_encoder.get_embd_files(trainer.num_devices)
            logger.info(f"Embedding files exist: {queries_embds_files}")
            dm.set_queries_embds(queries_embds_files=queries_embds_files)
        else:
            logger.info(f"in_memory = {rteb_encoder.in_memory}")
            logger.info(f"save_file = {rteb_encoder.save_file}")
            trainer.predict(model=rteb_encoder, dataloaders=dm.queries_dataloader())
            # Set the query embeddings
            queries_embds_files = rteb_encoder.get_embd_files()
            if rteb_encoder.in_memory:
                dm.set_queries_embds(queries_embds=rteb_encoder.embds)
            else:
                dm.set_queries_embds(queries_embds_files=queries_embds_files)
        trainer.strategy.barrier()  # Ensure embeddings are ready on all ranks

        # Encode Corpus
        logger.info("Encoding corpus")
        rteb_encoder.is_query = False
        rteb_encoder.in_memory = len(dm.dataset.corpus) < args.embd_in_memory_threshold
        rteb_encoder.save_file = str(corpus_embds_file)

        if args.load_embds and corpus_embds_file.exists():
            if trainer.is_global_zero:
                logger.info(f"Loading corpus embeddings from {corpus_embds_file}")
            dm.set_corpus_embds(
                corpus_embds_files=[str(corpus_embds_file)]
            )  # Pass as list
        else:
            if trainer.is_global_zero:
                logger.info(f"in_memory = {rteb_encoder.in_memory}")
                logger.info(f"save_file = {rteb_encoder.save_file}")
            trainer.predict(model=rteb_encoder, dataloaders=dm.corpus_dataloader())
            if rteb_encoder.in_memory:
                dm.set_corpus_embds(corpus_embds=rteb_encoder.embds)
            else:
                dm.set_corpus_embds(corpus_embds_files=[str(corpus_embds_file)])

        trainer.strategy.barrier()  # Ensure embeddings are ready on all ranks

        # 3. Manually Perform Retrieval
        logger.info("Retrieve")
        retriever_instance = Retriever(topk=100)  # Instantiate Retriever
        retriever_instance.corpus_embd_dataloader = dm.corpus_embd_dataloader()
        retriever_instance.in_memory = (
            len(dm.dataset.queries) < args.embd_in_memory_threshold
        )
        retriever_instance.save_file = str(
            task_save_path / RETRIEVE_PRED_FILENAME
        )  # Use consistent filename
        retriever_instance.save_prediction = True  # Ensure prediction is saved

        trainer.predict(
            model=retriever_instance, dataloaders=dm.queries_embd_dataloader()
        )

        # Remove the embeddings if not saving
        if not args.save_embds and not args.load_embds and trainer.is_global_zero:
            if queries_embds_file.exists():
                os.remove(queries_embds_file)
            if corpus_embds_file.exists():
                os.remove(corpus_embds_file)

        # 4. Run Evaluation
        rteb_scores = {}
        if trainer.is_global_zero:
            try:
                relevance_data = dm.dataset.relevance
                if not relevance_data:
                    logger.error("Ground truth relevance data not found or empty.")
                    raise ValueError("Relevance data is missing.")

                # Load predictions from the file saved by the retriever
                prediction_file = task_save_path / RETRIEVE_PRED_FILENAME
                if not prediction_file.exists():
                    logger.error(f"Prediction file not found at {prediction_file}")
                    raise FileNotFoundError(
                        f"Prediction file not found at {prediction_file}"
                    )

                with open(str(prediction_file)) as f:
                    predictions = json.load(f)

                filtered_predictions = {
                    qid: scores
                    for qid, scores in predictions.items()
                    if qid in relevance_data
                }
                if len(filtered_predictions) != len(relevance_data):
                    logger.warning(
                        f"Number of queries in predictions ({len(filtered_predictions)}) does not match relevance data ({len(relevance_data)}). Evaluating on intersection."
                    )
                    filtered_relevance = {
                        qid: scores
                        for qid, scores in relevance_data.items()
                        if qid in filtered_predictions
                    }
                else:
                    filtered_relevance = relevance_data

                if not filtered_predictions:
                    logger.error(
                        "No overlapping queries between predictions and relevance data."
                    )
                    raise ValueError("No queries to evaluate.")

                rteb_scores = run_retrieve_evaluation(
                    filtered_relevance, filtered_predictions
                )

                logger.info("-" * 40)
                logger.info(f"Dataset: {rteb_dataset_name}")
                logger.info(f"Model: {model_name}")
                logger.info(f"Save path: {task_save_path}")
                logger.info("Retrieval evaluation:")
                logger.info(rteb_scores)  # Log the scores dictionary

                # 5. Format and Save Results
                mteb_scores = dict(rteb_scores)
                if task_metadata.main_score not in mteb_scores:
                    logger.warning(
                        f"Main score '{task_metadata.main_score}' not found in RTEB results."
                    )
                    fallback_score = (
                        next(iter(mteb_scores.values()), 0.0) if mteb_scores else 0.0
                    )
                    mteb_scores["main_score"] = fallback_score
                else:
                    mteb_scores["main_score"] = mteb_scores[task_metadata.main_score]

                mteb_scores["model_name"] = model_name
                if rteb_encoder.embd_dim:
                    mteb_scores["embd_dim"] = rteb_encoder.embd_dim
                mteb_scores["embd_dtype"] = rteb_encoder.embd_dtype

                keys_to_remove = ["model_name", "embd_dim", "embd_dtype"]
                final_scores = {}
                for key, value in mteb_scores.items():
                    if key not in keys_to_remove:
                        try:
                            final_scores[key] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not convert score '{key}' to float. Skipping."
                            )

                if "main_score" not in final_scores and "main_score" in mteb_scores:
                    try:
                        final_scores["main_score"] = float(mteb_scores["main_score"])
                    except (ValueError, TypeError):
                        final_scores["main_score"] = 0.0

                final_scores["hf_subset"] = hf_subset if is_multilingual else "default"
                final_scores["languages"] = task_metadata.eval_langs

                with open(str(eval_file), "w") as f:
                    json.dump(final_scores, f)
                logger.info(f"Results saved to: {eval_file}")
                rteb_scores = final_scores  # Return the final formatted scores

            except Exception as e:
                logger.error(
                    f"Error during score calculation or saving: {e}", exc_info=True
                )
                rteb_scores = {
                    "main_score": 0.0,
                    task_metadata.main_score: 0.0,
                    "hf_subset": hf_subset if is_multilingual else "default",
                    "languages": task_metadata.eval_langs,
                }

        trainer.strategy.barrier()  # Ensure global zero finishes saving before other ranks proceeds

        # If not global zero, wait for global zero to save and then load the results
        if not trainer.is_global_zero:
            if eval_file.exists():
                with open(str(eval_file)) as f:
                    rteb_scores = json.load(f)
            else:
                logger.error(
                    f"Evaluation file not found on non-global zero rank: {eval_file}"
                )
                rteb_scores = {
                    "main_score": 0.0,
                    task_metadata.main_score: 0.0,
                    "hf_subset": hf_subset if is_multilingual else "default",
                    "languages": task_metadata.eval_langs,
                }

        logger.info(f"Finished RTEB evaluation for {task_metadata.name}.")
        return rteb_scores
