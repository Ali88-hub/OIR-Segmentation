"""Tests for config.py (RAG) and src/config.py (model)."""

from pathlib import Path

import pytest

from config import CHROMA_PATH, COLLECTION_NAME, DEFAULT_QUERIES, EMBEDDING_MODEL
from src.config import Config

# ── RAG config ────────────────────────────────────────────────────────────────


def test_rag_config_types():
    assert isinstance(EMBEDDING_MODEL, str) and EMBEDDING_MODEL
    assert isinstance(COLLECTION_NAME, str) and COLLECTION_NAME
    assert isinstance(CHROMA_PATH, Path)


def test_default_queries_non_empty():
    assert len(DEFAULT_QUERIES) > 0
    assert all(isinstance(q, str) and q.strip() for q in DEFAULT_QUERIES)


# ── Model config ──────────────────────────────────────────────────────────────


def test_config_defaults_valid():
    cfg = Config()
    assert cfg.num_classes == len(cfg.mask_names)
    assert cfg.num_classes == len(cfg.loss_weights)
    assert cfg.num_classes == len(cfg.tversky_alpha)
    assert cfg.num_classes == len(cfg.tversky_beta)


def test_config_rejects_mismatched_mask_names():
    with pytest.raises(AssertionError):
        Config(mask_names=("nv", "vo"), num_classes=3)


def test_config_rejects_mismatched_loss_weights():
    with pytest.raises(AssertionError):
        Config(loss_weights=(1.0, 1.0))  # 2 weights, 3 classes


def test_config_image_size_tuple():
    cfg = Config()
    assert len(cfg.image_size) == 2
    assert all(s > 0 for s in cfg.image_size)
