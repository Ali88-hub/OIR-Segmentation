"""Tests for pure utility functions in src/predict.py."""

import numpy as np

from src.predict import MAX_INPUT_SIZE, resize_to_max

# ── resize_to_max ─────────────────────────────────────────────────────────────


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_small_image_unchanged():
    img = _make_image(512, 512)
    out, scale = resize_to_max(img, max_side=1024)
    assert out.shape == (512, 512, 3)
    assert scale == 1.0


def test_exact_max_unchanged():
    img = _make_image(1024, 1024)
    out, scale = resize_to_max(img, max_side=1024)
    assert out.shape == (1024, 1024, 3)
    assert scale == 1.0


def test_wide_image_downscaled():
    img = _make_image(512, 2048)
    out, scale = resize_to_max(img, max_side=1024)
    assert out.shape[1] == 1024  # width capped at max_side
    assert out.shape[0] == 256  # height scaled proportionally
    assert abs(scale - 0.5) < 1e-6


def test_tall_image_downscaled():
    img = _make_image(2048, 512)
    out, scale = resize_to_max(img, max_side=1024)
    assert out.shape[0] == 1024
    assert out.shape[1] == 256
    assert abs(scale - 0.5) < 1e-6


def test_scale_maintains_aspect_ratio():
    img = _make_image(600, 800)
    out, scale = resize_to_max(img, max_side=400)
    assert abs(out.shape[0] / out.shape[1] - 600 / 800) < 0.02


def test_default_max_side_is_1024():
    img = _make_image(2000, 1000)
    out, scale = resize_to_max(img)
    assert max(out.shape[:2]) == MAX_INPUT_SIZE
