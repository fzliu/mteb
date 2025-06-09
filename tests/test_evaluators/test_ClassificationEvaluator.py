import pytest
import numpy as np
import torch
from mteb.evaluation.evaluators import (
    kNNClassificationEvaluator,
    kNNClassificationEvaluatorPytorch,
    logRegClassificationEvaluator,
)
from tests.test_benchmark.mock_models import MockNumpyEncoder

# Basic test data
SENTENCES_TRAIN_BINARY = [
    "this is a positive sentence",
    "another positive sentence",
    "this is a negative sentence",
    "another negative sentence",
]
Y_TRAIN_BINARY = np.array([1, 1, 0, 0])
SENTENCES_TEST_BINARY = [
    "a new positive sentence",
    "a new negative sentence",
]
Y_TEST_BINARY = np.array([1, 0])

SENTENCES_TRAIN_MULTICLASS = [
    "class 0 sentence 1",
    "class 0 sentence 2",
    "class 1 sentence 1",
    "class 1 sentence 2",
    "class 2 sentence 1",
    "class 2 sentence 2",
]
Y_TRAIN_MULTICLASS = np.array([0, 0, 1, 1, 2, 2])
SENTENCES_TEST_MULTICLASS = [
    "new class 0 sentence",
    "new class 1 sentence",
    "new class 2 sentence",
]
Y_TEST_MULTICLASS = np.array([0, 1, 2])

# For checking the cache
SENTENCES_TEST_CACHE = [
    "another new positive sentence",
    "another new negative sentence",
]
Y_TEST_CACHE = np.array([1, 0])


class TestKNNClassificationEvaluator:
    def setup_method(self):
        self.model = MockNumpyEncoder()
        self.eval_binary = kNNClassificationEvaluator(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_knn_binary"
        )
        self.eval_multiclass = kNNClassificationEvaluator(
            SENTENCES_TRAIN_MULTICLASS, Y_TRAIN_MULTICLASS, SENTENCES_TEST_MULTICLASS, Y_TEST_MULTICLASS, task_name="test_knn_multiclass"
        )

    def test_output_structure_binary(self):
        scores, test_cache = self.eval_binary(self.model)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, np.ndarray)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "ap" in scores  # Binary classification should have AP
        assert "accuracy_cosine" in scores
        assert "f1_cosine" in scores
        assert "ap_cosine" in scores
        assert "accuracy_euclidean" in scores
        assert "f1_euclidean" in scores
        assert "ap_euclidean" in scores

    def test_output_structure_multiclass(self):
        scores, test_cache = self.eval_multiclass(self.model)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, np.ndarray)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "ap" not in scores  # Multiclass classification should not have AP by default
        assert "accuracy_cosine" in scores
        assert "f1_cosine" in scores
        assert "accuracy_euclidean" in scores
        assert "f1_euclidean" in scores

    def test_score_ranges_binary(self):
        scores, _ = self.eval_binary(self.model)
        for metric in ["accuracy", "f1", "ap", "accuracy_cosine", "f1_cosine", "ap_cosine", "accuracy_euclidean", "f1_euclidean", "ap_euclidean"]:
            assert 0 <= scores[metric] <= 1

    def test_score_ranges_multiclass(self):
        scores, _ = self.eval_multiclass(self.model)
        for metric in ["accuracy", "f1", "accuracy_cosine", "f1_cosine", "accuracy_euclidean", "f1_euclidean"]:
            assert 0 <= scores[metric] <= 1

    def test_cache_usage_binary(self):
        _, test_cache_initial = self.eval_binary(self.model)
        # Create a new evaluator for the second call to ensure fresh state apart from cache
        eval_binary_cache_test = kNNClassificationEvaluator(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_knn_binary_cache"
        )
        scores_with_cache, test_cache_after_cache_usage = eval_binary_cache_test(self.model, test_cache=test_cache_initial)

        # Check that the cached embeddings are used (mock encoder would be called otherwise)
        # For MockNumpyEncoder, encode is called, but the idea is that X_test is not recomputed
        # We can't directly check if encode was called less without more advanced mocking
        assert np.array_equal(test_cache_initial, test_cache_after_cache_usage)
        for metric in ["accuracy", "f1", "ap"]:
            assert 0 <= scores_with_cache[metric] <= 1

    def test_limit_parameter(self):
        eval_limited = kNNClassificationEvaluator(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_knn_limit", limit=2
        )
        # Check that the data is actually limited
        assert len(eval_limited.sentences_train) == 2
        assert len(eval_limited.y_train) == 2
        assert len(eval_limited.sentences_test) == 2 # Test data also gets limited
        assert len(eval_limited.y_test) == 2

        scores, _ = eval_limited(self.model)
        assert "accuracy" in scores
        assert "f1" in scores
        # AP might not be meaningful with 2 test samples if they become same class after limit
        # but the key should exist if it's binary
        if len(np.unique(eval_limited.y_train)) == 2 and len(np.unique(eval_limited.y_test)) == 2 :
             assert "ap" in scores

        # Try with a limit that makes it not binary anymore for y_train to check ap exclusion
        eval_limited_not_binary_train = kNNClassificationEvaluator(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_knn_limit_not_binary", limit=1 # Only 1 training sample
        )
        scores_not_binary, _ = eval_limited_not_binary_train(self.model)
        if len(np.unique(eval_limited_not_binary_train.y_train)) < 2:
            assert "ap" not in scores_not_binary


class TestKNNClassificationEvaluatorPytorch:
    def setup_method(self):
        self.model = MockNumpyEncoder(is_pytorch_model=True) # Assuming MockNumpyEncoder can pretend to be a PyTorch model
        self.eval_binary = kNNClassificationEvaluatorPytorch(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_knn_pytorch_binary"
        )
        self.eval_multiclass = kNNClassificationEvaluatorPytorch(
            SENTENCES_TRAIN_MULTICLASS, Y_TRAIN_MULTICLASS, SENTENCES_TEST_MULTICLASS, Y_TEST_MULTICLASS, task_name="test_knn_pytorch_multiclass"
        )

    def test_output_structure_binary(self):
        scores, test_cache = self.eval_binary(self.model)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, torch.Tensor) # Pytorch version should return torch.Tensor
        assert "accuracy" in scores
        assert "f1" in scores
        assert "ap" in scores
        assert "accuracy_cosine" in scores
        assert "f1_cosine" in scores
        assert "ap_cosine" in scores
        assert "accuracy_euclidean" in scores
        assert "f1_euclidean" in scores
        assert "ap_euclidean" in scores
        assert "accuracy_dot" in scores
        assert "f1_dot" in scores
        assert "ap_dot" in scores

    def test_output_structure_multiclass(self):
        scores, test_cache = self.eval_multiclass(self.model)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, torch.Tensor)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "ap" not in scores
        assert "accuracy_cosine" in scores
        assert "f1_cosine" in scores
        assert "accuracy_euclidean" in scores
        assert "f1_euclidean" in scores
        assert "accuracy_dot" in scores
        assert "f1_dot" in scores

    def test_score_ranges_binary(self):
        scores, _ = self.eval_binary(self.model)
        for metric_key in scores.keys():
            if "accuracy" in metric_key or "f1" in metric_key or "ap" in metric_key:
                 assert 0 <= scores[metric_key] <= 1

    def test_score_ranges_multiclass(self):
        scores, _ = self.eval_multiclass(self.model)
        for metric_key in scores.keys():
            if "accuracy" in metric_key or "f1" in metric_key:
                assert 0 <= scores[metric_key] <= 1

    def test_cache_usage_binary(self):
        _, test_cache_initial = self.eval_binary(self.model)
        # Create a new evaluator for the second call
        eval_binary_cache_test = kNNClassificationEvaluatorPytorch(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_knn_pytorch_binary_cache"
        )
        scores_with_cache, test_cache_after_cache_usage = eval_binary_cache_test(self.model, test_cache=test_cache_initial)

        assert torch.equal(test_cache_initial, test_cache_after_cache_usage)
        for metric_key in scores_with_cache.keys():
            if "accuracy" in metric_key or "f1" in metric_key or "ap" in metric_key:
                 assert 0 <= scores_with_cache[metric_key] <= 1

    def test_limit_parameter(self):
        eval_limited = kNNClassificationEvaluatorPytorch(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_knn_pytorch_limit", limit=2
        )
        assert len(eval_limited.sentences_train) == 2
        assert len(eval_limited.y_train) == 2
        assert len(eval_limited.sentences_test) == 2
        assert len(eval_limited.y_test) == 2

        scores, _ = eval_limited(self.model)
        assert "accuracy" in scores
        assert "f1" in scores
        if len(np.unique(eval_limited.y_train)) == 2 and len(np.unique(eval_limited.y_test)) == 2:
             assert "ap" in scores

        eval_limited_not_binary_train = kNNClassificationEvaluatorPytorch(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_knn_pytorch_limit_not_binary", limit=1
        )
        scores_not_binary, _ = eval_limited_not_binary_train(self.model)
        if len(np.unique(eval_limited_not_binary_train.y_train)) < 2:
            assert "ap" not in scores_not_binary


class TestLogRegClassificationEvaluator:
    def setup_method(self):
        self.model = MockNumpyEncoder()
        self.eval_binary = logRegClassificationEvaluator(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_logreg_binary"
        )
        self.eval_multiclass = logRegClassificationEvaluator(
            SENTENCES_TRAIN_MULTICLASS, Y_TRAIN_MULTICLASS, SENTENCES_TEST_MULTICLASS, Y_TEST_MULTICLASS, task_name="test_logreg_multiclass"
        )

    def test_output_structure_binary(self):
        scores, test_cache = self.eval_binary(self.model)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, np.ndarray)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "f1_weighted" in scores
        assert "ap" in scores  # Binary classification should have AP
        assert "ap_weighted" in scores

    def test_output_structure_multiclass(self):
        scores, test_cache = self.eval_multiclass(self.model)
        assert isinstance(scores, dict)
        assert isinstance(test_cache, np.ndarray)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "f1_weighted" in scores
        assert "ap" not in scores  # Multiclass classification should not have AP by default (only macro and weighted)

    def test_score_ranges_binary(self):
        scores, _ = self.eval_binary(self.model)
        for metric in ["accuracy", "f1", "f1_weighted", "ap", "ap_weighted"]:
            assert 0 <= scores[metric] <= 1

    def test_score_ranges_multiclass(self):
        scores, _ = self.eval_multiclass(self.model)
        for metric in ["accuracy", "f1", "f1_weighted"]:
            assert 0 <= scores[metric] <= 1

    def test_cache_usage_binary(self):
        _, test_cache_initial = self.eval_binary(self.model)
        # Create a new evaluator for the second call
        eval_binary_cache_test = logRegClassificationEvaluator(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_logreg_binary_cache"
        )
        scores_with_cache, test_cache_after_cache_usage = eval_binary_cache_test(self.model, test_cache=test_cache_initial)

        assert np.array_equal(test_cache_initial, test_cache_after_cache_usage)
        for metric in ["accuracy", "f1", "f1_weighted", "ap", "ap_weighted"]:
            assert 0 <= scores_with_cache[metric] <= 1

    def test_limit_parameter(self):
        # Test with a limit that keeps it binary
        eval_limited_binary = logRegClassificationEvaluator(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_logreg_limit_binary", limit=2
        )
        assert len(eval_limited_binary.sentences_train) == 2
        assert len(eval_limited_binary.y_train) == 2
        assert len(eval_limited_binary.sentences_test) == 2
        assert len(eval_limited_binary.y_test) == 2

        scores_limited_binary, _ = eval_limited_binary(self.model)
        assert "accuracy" in scores_limited_binary
        assert "f1" in scores_limited_binary
        # Ensure 'ap' is present if still binary after limiting
        if len(np.unique(eval_limited_binary.y_train)) == 2 and len(np.unique(eval_limited_binary.y_test)) == 2:
            assert "ap" in scores_limited_binary
            assert "ap_weighted" in scores_limited_binary
        else: # if limiting made it not binary (e.g. all one class)
            assert "ap" not in scores_limited_binary
            assert "ap_weighted" not in scores_limited_binary


        # Test with a limit that makes y_train not binary
        eval_limited_not_binary_train = logRegClassificationEvaluator(
            SENTENCES_TRAIN_BINARY, Y_TRAIN_BINARY, SENTENCES_TEST_BINARY, Y_TEST_BINARY, task_name="test_logreg_limit_not_binary_train", limit=1
        )
        scores_not_binary_train, _ = eval_limited_not_binary_train(self.model)
        # If training data is not binary, 'ap' should not be calculated
        if len(np.unique(eval_limited_not_binary_train.y_train)) < 2:
            assert "ap" not in scores_not_binary_train
            assert "ap_weighted" not in scores_not_binary_train
