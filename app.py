"""
OIRseg Streamlit App — Local Prediction Interface

Run:
    streamlit run app.py
"""

import warnings

warnings.filterwarnings("ignore", message="Accessing `__path__`")

import csv
import io
import os
import re
import shutil
import zipfile
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.config import Config
from src.predict import (
    MAX_INPUT_SIZE,
    get_preprocess,
    load_vessel_mask,
    postprocess_all,
    predict_single,
)
from theme import inject_theme, page_header, section_header, sidebar_legend, sidebar_status

# ── RAG (lazy — only imported when user accesses RAG features) ────────────────
_rag = None


def _ensure_rag():
    """Lazily import the rag module on first use to avoid loading
    sentence-transformers/transformers at startup (saves ~500 MB RAM)."""
    global _rag
    if _rag is None:
        import rag as _rag_mod

        _rag = _rag_mod
    return _rag


@st.cache_data(show_spinner=False)
def _rag_available() -> bool:
    """Check if RAG dependencies are installed without importing them."""
    import importlib.util

    for mod in ("sentence_transformers", "anthropic", "chromadb"):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


@st.cache_resource(show_spinner="Loading PubMed index…")
def _get_rag_retriever():
    mod = _ensure_rag()
    return mod._load_retriever()


# ── Filename parser ────────────────────────────────────────────────────────────
def parse_filename(name: str) -> dict:
    """Parse OIR experiment metadata from the image filename (best-effort).

    Expected structure (underscore-separated):
      [pred_{idx}_]{birth_date}_{experiment}_{image_date}_{protocol}_{animal_info}

    Falls back gracefully for single-date or non-standard filenames.
    Returns a dict with whatever fields could be extracted, always including 'raw'.
    """
    stem = Path(name).stem
    result: dict = {"raw": name}

    stem = re.sub(r"^pred_\d+_", "", stem)

    date_pat = r"\d{4}\.\d{2}\.\d{2}"
    dates = re.findall(date_pat, stem)

    if len(dates) >= 1:
        result["birth_date"] = dates[0]
    if len(dates) >= 2:
        result["image_date"] = dates[1]
        try:
            d1 = datetime.strptime(dates[0], "%Y.%m.%d")
            d2 = datetime.strptime(dates[1], "%Y.%m.%d")
            result["postnatal_day"] = (d2 - d1).days
        except ValueError:
            pass

    # Split stem into segments using dates as anchors
    # Result: [before_date1, date1, between_dates, date2, after_date2, ...]
    segments = [s.strip("_").strip() for s in re.split(date_pat, stem) if s.strip("_").strip()]

    if len(dates) >= 2:
        # segments: [experiment_name, protocol+animal]
        if len(segments) >= 1:
            result["experiment"] = segments[0]
        rest = "_".join(segments[1:]) if len(segments) > 1 else ""
    elif len(dates) == 1:
        # Only one date — everything after it is protocol+animal
        after_date = re.split(re.escape(dates[0]), stem, maxsplit=1)[-1].lstrip("_").strip()
        before_date = re.split(re.escape(dates[0]), stem, maxsplit=1)[0].strip("_").strip()
        if before_date:
            result["experiment"] = before_date
        rest = after_date
    else:
        # No dates — treat the whole stem as a fallback, nothing more to parse
        rest = ""

    if rest:
        animal_match = re.search(r"(Litter\s*\d|[Ll]\d[\s_]+[Mm]\d|[Mm]\d+[LR]\b)", rest)
        if animal_match:
            protocol = rest[: animal_match.start()].strip("_").strip()
            animal = rest[animal_match.start() :].strip()
        else:
            parts = rest.split("_")
            protocol = "_".join(parts[:-1]).strip()
            animal = parts[-1].strip()

        if protocol:
            result["protocol"] = protocol
        if animal:
            result["animal_info"] = animal
            eye_m = re.search(r"[Mm]\d+([LR])\b", animal)
            if eye_m:
                result["eye"] = "Left" if eye_m.group(1) == "L" else "Right"
            tokens = animal.replace("_", " ").split()
            if tokens:
                result["treatment"] = tokens[-1]
            litter_m = re.search(r"[Ll]itter\s*(\d+)|[Ll](\d+)[\s_]*[Mm]", animal)
            if litter_m:
                result["litter"] = litter_m.group(1) or litter_m.group(2)

    return result


# ── Constants ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = str(_PROJECT_ROOT / "Model V4 output" / "checkpoints" / "best_model.pth")
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500 MP cap; allow large retinal flatmounts
MASK_COLORS = {
    "nv": (1.0, 0.0, 0.0),
    "vo": (1.0, 1.0, 1.0),
    "retina": (0.3, 0.3, 1.0),
    "background": (0.3, 0.3, 1.0),
}
_MAX_RAG_MESSAGES = 50


def _render_sources(sources: list[dict], title: str = "Sources") -> None:
    """Render a list of literature sources inside an expander."""
    if not sources:
        return
    with st.expander(f"{title} ({len(sources)})"):
        for src in sources:
            label = (
                f"[{src['ref']}]  {src['title']}  ({src['year']})"
                if src["year"]
                else f"[{src['ref']}]  {src['title']}"
            )
            st.markdown(f"**{label}**")
            if src["authors"]:
                st.caption(f"{src['authors']} · *{src['journal']}* · score: `{src['score']}`")
            if src["url"]:
                st.link_button("Open in PubMed", src["url"])


MASK_LABELS = {
    "nv": "Neovascular",
    "vo": "Vaso-Obliterated",
    "retina": "Retina",
    "background": "Retina",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OIRseg — Retinal Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()

page_header(
    title="OIRseg",
    subtitle="Retinal Flatmount Segmentation",
    caption="U-Net + EfficientNet-B4  ·  scSE Attention  ·  Neovascular / Vaso-Obliterated / Background",
)


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_model(ckpt_mtime: float):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    if "config" in ckpt:
        saved = ckpt["config"]
        config.image_size = tuple(saved.get("image_size", config.image_size))
        config.encoder_name = saved.get("encoder_name", config.encoder_name)
        config.num_classes = saved.get("num_classes", config.num_classes)
        config.mask_names = tuple(saved.get("mask_names", config.mask_names))
    # Derive decoder_attention from actual weights — the saved config value can
    # lag behind what was really trained (e.g. config default said "scse" but
    # the model was built without it).
    has_scse = any("sSE" in k or "cSE" in k for k in state.keys())
    config.decoder_attention = "scse" if has_scse else None
    from src.model import build_model

    model = build_model(config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    preprocess = get_preprocess(config)
    return model, preprocess, config, device


if not Path(CHECKPOINT).exists():
    with st.spinner("Downloading model from Hugging Face Hub…"):
        try:
            from download_model import ensure_checkpoint

            ensure_checkpoint()
        except Exception as e:
            st.error(
                f"Checkpoint not found: `{CHECKPOINT}`\n\n"
                f"Auto-download failed: {e}\n\n"
                f"Place `best_model.pth` at `Model V4 output/checkpoints/` manually."
            )
            st.stop()

model, preprocess, config, device = get_model(os.path.getmtime(CHECKPOINT))
sidebar_status(f"Model loaded  ·  {device}")

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
<h3 style="font-family:'Playfair Display',serif;color:#C5A55A;margin-bottom:0.5rem;">Settings</h3>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
<p style="color:#9A8B6F;font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.3rem;">Thresholds</p>
""",
    unsafe_allow_html=True,
)

thresh_nv = st.sidebar.number_input(
    "NV — Neovascular",
    0.01,
    0.99,
    0.69,
    0.01,
    format="%.2f",
    help="Recommended 0.65–0.71 · optimal 0.69 (V4)",
)
thresh_vo = st.sidebar.number_input(
    "VO — Vaso-Obliterated",
    0.01,
    0.99,
    0.99,
    0.01,
    format="%.2f",
    help="Recommended 0.55–0.63 · optimal 0.59 (V4)",
)
thresh_bg = st.sidebar.number_input(
    "BG — Retina",
    0.01,
    0.99,
    0.63,
    0.01,
    format="%.2f",
    help="Recommended 0.63–0.69 · optimal 0.67 (V4)",
)
thresholds = {"nv": thresh_nv, "vo": thresh_vo, "retina": thresh_bg, "background": thresh_bg}

st.sidebar.markdown(
    """
<div style="background:rgba(197,165,90,0.07);border:1px solid rgba(197,165,90,0.15);
            border-radius:7px;padding:0.4rem 0.65rem;margin-top:0.2rem;">
  <span style="color:#9A8B6F;font-size:0.68rem;letter-spacing:0.06em;text-transform:uppercase;">rec:&nbsp;</span>
  <span style="color:#BFB39A;font-size:0.68rem;">NV 0.65–0.71&nbsp;·&nbsp;VO 0.55–0.63&nbsp;·&nbsp;BG 0.63–0.69</span>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """<div style="height:1px;margin:0.8rem 0;background:linear-gradient(90deg,rgba(197,165,90,0.4),rgba(197,165,90,0.05),transparent);"></div>""",
    unsafe_allow_html=True,
)

tta = st.sidebar.checkbox(
    "Test-Time Augmentation (TTA)",
    value=True,
    help="Average predictions over flips — slower but more accurate",
)
show_prob = st.sidebar.checkbox("Show probability maps", value=False)

st.sidebar.markdown(
    """<div style="height:1px;margin:0.8rem 0;background:linear-gradient(90deg,rgba(197,165,90,0.4),rgba(197,165,90,0.05),transparent);"></div>""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
<p style="font-family:'Playfair Display',serif;color:#E8D5A3;font-size:0.95rem;font-weight:600;margin-bottom:0.4rem;">Mask Colours</p>
""",
    unsafe_allow_html=True,
)

sidebar_legend(
    [
        ("NV", "Neovascularization", MASK_COLORS["nv"]),
        ("VO", "Vascular Obliteration", MASK_COLORS["vo"]),
        ("BG", "Retina", MASK_COLORS["retina"]),
    ]
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_lit = st.tabs(["Single Image", "Batch", "Literature"])


# ═════════════════════════════════════════════════════════════════════════════
#  SINGLE IMAGE
# ═════════════════════════════════════════════════════════════════════════════
with tab_single:
    uploaded = st.file_uploader(
        "Upload a retinal flatmount image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=False,
        key="single_upload",
    )

    image_pil = None
    if uploaded is not None:
        try:
            image_pil = Image.open(uploaded).convert("RGB")
        except (Image.UnidentifiedImageError, OSError) as e:
            st.error(f"Could not open image: {e}")

    if image_pil is not None:
        image_np = np.array(image_pil)
        orig_h, orig_w = image_np.shape[:2]

        col_img, col_space, col_info = st.columns([2, 2, 1])
        with col_img:
            st.image(image_pil, caption=f"Input: {uploaded.name}", width="stretch")
        with col_info:
            st.metric("Width", f"{orig_w} px")
            st.metric("Height", f"{orig_h} px")
            if orig_h > MAX_INPUT_SIZE or orig_w > MAX_INPUT_SIZE:
                st.warning(f"Image exceeds {MAX_INPUT_SIZE}px — will be downscaled.")

        run = st.button(
            "Run Segmentation", type="primary", use_container_width=True, key="run_single"
        )

        if run:
            _infer_placeholder = st.empty()
            _infer_placeholder.markdown(
                """
<div style="display:flex;justify-content:center;padding:0.6rem 0 0.4rem 0;">
<svg width="220" height="190" viewBox="0 0 220 190" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="inf_wheelGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%"   stop-color="#C5A55A" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="#C5A55A" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="inf_goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#E8D5A3"/>
      <stop offset="100%" stop-color="#8B7635"/>
    </linearGradient>
    <radialGradient id="inf_bodyGrad" cx="50%" cy="30%" r="60%">
      <stop offset="0%"   stop-color="#CFC4AA"/>
      <stop offset="100%" stop-color="#8A7B62"/>
    </radialGradient>
    <radialGradient id="inf_bellyGrad" cx="50%" cy="60%" r="55%">
      <stop offset="0%"   stop-color="#E0D5BC"/>
      <stop offset="100%" stop-color="#B0A282"/>
    </radialGradient>
  </defs>
  <circle cx="110" cy="78" r="68" fill="url(#inf_wheelGlow)"/>
  <g>
    <animateTransform attributeName="transform" type="rotate"
      from="0 110 78" to="360 110 78" dur="1.3s" repeatCount="indefinite"/>
    <circle cx="110" cy="78" r="60" fill="none" stroke="#C5A55A" stroke-width="4.5" opacity="0.8"/>
    <circle cx="110" cy="78" r="60" fill="none" stroke="#8B7635"
            stroke-width="2" stroke-dasharray="5 8" opacity="0.3"/>
    <circle cx="110" cy="78" r="52" fill="none" stroke="#8B7635" stroke-width="1.2" opacity="0.3"/>
    <circle cx="110" cy="78" r="8"  fill="#C5A55A" opacity="0.9"/>
    <circle cx="110" cy="78" r="4"  fill="#1a1612"/>
    <g stroke="url(#inf_goldGrad)" stroke-width="2.2" opacity="0.6" stroke-linecap="round">
      <line x1="110" y1="70"  x2="110" y2="26"/>
      <line x1="110" y1="86"  x2="110" y2="130"/>
      <line x1="102" y1="78"  x2="58"  y2="78"/>
      <line x1="118" y1="78"  x2="162" y2="78"/>
      <line x1="104" y1="72"  x2="74"  y2="42"/>
      <line x1="116" y1="84"  x2="146" y2="114"/>
      <line x1="116" y1="72"  x2="146" y2="42"/>
      <line x1="104" y1="84"  x2="74"  y2="114"/>
    </g>
  </g>
  <g>
    <animateTransform attributeName="transform" type="translate"
      values="0,0; 0,-1.5; 0,0; 0,-1.5; 0,0"
      dur="0.5s" repeatCount="indefinite"/>
    <g transform="translate(117,117) scale(1.2)">
      <path fill="none" stroke="#9A8B6F" stroke-width="2" stroke-linecap="round">
        <animate attributeName="d"
          values="M -15 2  C -25  5  -33 -2  -31 -11  Q -29 -17 -24 -15;
                  M -15 2  C -23 -1  -30 -10  -27 -17  Q -24 -22 -19 -19;
                  M -15 2  C -25  5  -33 -2  -31 -11  Q -29 -17 -24 -15"
          dur="0.5s" repeatCount="indefinite"/>
      </path>
      <g opacity="0.65">
        <line stroke="#7A6B52" stroke-width="2.8" stroke-linecap="round" x1="-7" y1="8">
          <animate attributeName="x2" values="-13;-4;-13" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="11;9;11"    dur="0.5s" repeatCount="indefinite"/>
        </line>
        <line stroke="#7A6B52" stroke-width="2.2" stroke-linecap="round">
          <animate attributeName="x1" values="-13;-4;-13" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y1" values="8;6;8"      dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="x2" values="-16;-5;-16" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="13;10;13"   dur="0.5s" repeatCount="indefinite"/>
        </line>
        <ellipse rx="3.2" ry="1.6" fill="#7A6B52">
          <animate attributeName="cx" values="-16;-5;-16" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="cy" values="13;10;13"   dur="0.5s" repeatCount="indefinite"/>
        </ellipse>
      </g>
      <g opacity="0.65">
        <line stroke="#7A6B52" stroke-width="2.5" stroke-linecap="round" x1="6" y1="8">
          <animate attributeName="x2" values="1;10;1"  dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="11;9;11" dur="0.5s" repeatCount="indefinite"/>
        </line>
        <line stroke="#7A6B52" stroke-width="2" stroke-linecap="round">
          <animate attributeName="x1" values="1;10;1"   dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y1" values="8;6;8"    dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="x2" values="-1;13;-1" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="13;10;13" dur="0.5s" repeatCount="indefinite"/>
        </line>
        <ellipse rx="2.8" ry="1.5" fill="#7A6B52">
          <animate attributeName="cx" values="-1;13;-1" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="cy" values="13;10;13" dur="0.5s" repeatCount="indefinite"/>
        </ellipse>
      </g>
      <ellipse cx="-2" cy="-1" rx="18" ry="11" fill="url(#inf_bodyGrad)"/>
      <ellipse cx="-1" cy="7"  rx="13" ry="4"  fill="url(#inf_bellyGrad)" opacity="0.55"/>
      <g>
        <line stroke="#9A8B70" stroke-width="2.8" stroke-linecap="round" x1="-7" y1="8">
          <animate attributeName="x2" values="-4;-13;-4" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="9;11;9"    dur="0.5s" repeatCount="indefinite"/>
        </line>
        <line stroke="#9A8B70" stroke-width="2.2" stroke-linecap="round">
          <animate attributeName="x1" values="-4;-13;-4" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y1" values="6;8;6"     dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="x2" values="-5;-16;-5" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="10;13;10"  dur="0.5s" repeatCount="indefinite"/>
        </line>
        <ellipse rx="3.2" ry="1.6" fill="#BFB39A">
          <animate attributeName="cx" values="-5;-16;-5" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="cy" values="10;13;10"  dur="0.5s" repeatCount="indefinite"/>
        </ellipse>
      </g>
      <g>
        <line stroke="#9A8B70" stroke-width="2.5" stroke-linecap="round" x1="6" y1="8">
          <animate attributeName="x2" values="10;1;10" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="9;11;9"  dur="0.5s" repeatCount="indefinite"/>
        </line>
        <line stroke="#9A8B70" stroke-width="2" stroke-linecap="round">
          <animate attributeName="x1" values="10;1;10"  dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y1" values="6;8;6"    dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="x2" values="13;-1;13" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="10;13;10" dur="0.5s" repeatCount="indefinite"/>
        </line>
        <ellipse rx="2.8" ry="1.5" fill="#BFB39A">
          <animate attributeName="cx" values="13;-1;13" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="cy" values="10;13;10" dur="0.5s" repeatCount="indefinite"/>
        </ellipse>
      </g>
      <ellipse cx="12" cy="-5" rx="5.5" ry="4" fill="url(#inf_bodyGrad)"/>
      <circle cx="18" cy="-8" r="8.5" fill="url(#inf_bodyGrad)"/>
      <ellipse cx="14" cy="-16" rx="4.5" ry="5.5" fill="url(#inf_bodyGrad)"/>
      <ellipse cx="14" cy="-16" rx="2.8" ry="3.8" fill="#C5A55A" opacity="0.25"/>
      <circle cx="22" cy="-10" r="2.5" fill="#1a1612"/>
      <circle cx="23" cy="-11" r="0.9" fill="#E8D5A3" opacity="0.9"/>
      <ellipse cx="26" cy="-7" rx="1.8" ry="1.3" fill="#C5A55A" opacity="0.6"/>
      <g stroke="#9A8B6F" stroke-width="0.75" opacity="0.6" stroke-linecap="round">
        <line x1="26" y1="-8"  x2="35" y2="-10"/>
        <line x1="26" y1="-7"  x2="35" y2="-7"/>
        <line x1="26" y1="-6"  x2="35" y2="-4"/>
      </g>
    </g>
  </g>
  <ellipse cx="110" cy="142" rx="40" ry="3.5" fill="#C5A55A" opacity="0.07"/>
  <text x="110" y="157" text-anchor="middle"
        font-family="Inter,sans-serif" font-size="9" fill="#5A5345"
        letter-spacing="2.5">RUNNING INFERENCE</text>
</svg>
</div>
""",
                unsafe_allow_html=True,
            )
            try:
                masks_prob, _ = predict_single(
                    model,
                    image_np,
                    preprocess,
                    device,
                    config,
                    tta=tta,
                    threshold=0.5,
                )
            except Exception as e:
                st.error(f"Segmentation failed: {e}")
                st.stop()
            finally:
                _infer_placeholder.empty()
            # Apply per-class thresholds, then post-process all masks
            raw = [
                (masks_prob[i] > thresholds[name]).astype(np.uint8)
                for i, name in enumerate(config.mask_names)
            ]
            stem = Path(uploaded.name).stem
            vessel_mask = load_vessel_mask(stem, config.manifest_path, config.data_root)
            masks_binary = postprocess_all(
                np.stack(raw), config.mask_names, vessel_mask=vessel_mask, config=config
            )

            # Denominator = total retinal area (Connor et al. 2009 standard).
            # "retina" = retinal tissue; union of all masks = retinal area.
            retinal_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for _m in masks_binary:
                retinal_mask |= _m
            total_pixels = int(retinal_mask.sum()) or orig_h * orig_w

            section_header("Results")
            cols = st.columns(len(config.mask_names))
            seg_metrics: dict = {}
            for i, (col, name) in enumerate(zip(cols, config.mask_names)):
                px = int(masks_binary[i].sum())
                pct = round(100 * px / total_pixels, 4)
                seg_metrics[f"{name}_pct"] = pct
                col.metric(
                    label=MASK_LABELS.get(name, name.upper()),
                    value=f"{pct:.2f}%",
                    delta=f"{px:,} px",
                )

            # Store results in session state so the explanation section can access them
            st.session_state["single_seg_results"] = {
                "metrics": seg_metrics,
                "filename": uploaded.name,
            }

            section_header("Mask Overlays")
            overlay_cols = st.columns(len(config.mask_names))
            overlay_bufs = {}
            # Grayscale base from red channel (fluorescence images are nearly all red)
            gray = image_np[..., 0].astype(np.float32)
            base = np.stack([gray, gray, gray], axis=-1) / 255.0
            for i, (col, name) in enumerate(zip(overlay_cols, config.mask_names)):
                color = MASK_COLORS.get(name, (1.0, 1.0, 0.0))
                mask = masks_binary[i].astype(bool)
                overlay = base.copy()
                if name in ("nv", "vo"):
                    # Solid fill for NV (red) and VO (white)
                    overlay[mask] = list(color)
                else:
                    # Alpha blend for background/retina
                    alpha = mask.astype(np.float32) * 0.55
                    for c, cv in enumerate(color):
                        overlay[..., c] = base[..., c] * (1 - alpha) + cv * alpha
                overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
                col.image(
                    overlay_uint8,
                    caption=MASK_LABELS.get(name, name.upper()),
                    width="stretch",
                )
                buf = io.BytesIO()
                Image.fromarray(overlay_uint8).save(buf, format="PNG")
                overlay_bufs[name] = buf.getvalue()

            if show_prob:
                section_header("Probability Maps")
                prob_cols = st.columns(len(config.mask_names))
                for i, (col, name) in enumerate(zip(prob_cols, config.mask_names)):
                    col.image(
                        (masks_prob[i] * 255).astype(np.uint8),
                        caption=f"{MASK_LABELS.get(name, name.upper())} prob",
                        width="stretch",
                    )

            section_header("Download")
            stem = Path(uploaded.name).stem

            # Build all mask buffers once
            mask_bufs = {}
            for i, name in enumerate(config.mask_names):
                mask_pil = Image.fromarray(masks_binary[i] * 255)
                buf = io.BytesIO()
                mask_pil.save(buf, format="PNG")
                mask_bufs[name] = buf.getvalue()

            # Download all as ZIP (masks + overlays)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in mask_bufs.items():
                    zf.writestr(f"{stem}_{name}_mask.png", data)
                for name, data in overlay_bufs.items():
                    zf.writestr(f"{stem}_{name}_overlay.png", data)
            st.download_button(
                label="Download All (Masks + Overlays ZIP)",
                data=zip_buf.getvalue(),
                file_name=f"{stem}_masks.zip",
                mime="application/zip",
                use_container_width=True,
            )

        # ── AI Interpretation (shown after segmentation has run) ──────────────
        saved = st.session_state.get("single_seg_results")
        if saved and saved["filename"] == uploaded.name:
            section_header("AI Interpretation")

            if not _rag_available():
                st.warning(
                    "RAG module not available — install dependencies to enable AI interpretation."
                )
            else:
                # Show parsed experiment metadata
                meta = parse_filename(uploaded.name)
                meta_fields = {
                    "Experiment": meta.get("experiment"),
                    "Birth date": meta.get("birth_date"),
                    "Image date": meta.get("image_date"),
                    "Postnatal day (P)": meta.get("postnatal_day"),
                    "Protocol": meta.get("protocol"),
                    "Litter": meta.get("litter"),
                    "Eye": meta.get("eye"),
                    "Treatment": meta.get("treatment"),
                }
                filled = {k: v for k, v in meta_fields.items() if v is not None}
                if filled:
                    with st.expander("Parsed experiment metadata", expanded=True):
                        for k, v in filled.items():
                            st.markdown(
                                f'<span style="color:#9A8B6F;font-size:0.8rem;">{html_escape(str(k))}:</span> '
                                f'<span style="color:#E8D5A3;">{html_escape(str(v))}</span>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption(
                        "No structured metadata found in filename — interpretation will use raw name only."
                    )

                if st.button(
                    "Explain Results with AI",
                    type="primary",
                    use_container_width=True,
                    key="explain_single",
                ):
                    try:
                        _get_rag_retriever()  # ensure retriever is loaded
                        sources, stream = _ensure_rag().explain_segmentation(saved["metrics"], meta)
                    except RuntimeError as err:
                        st.error(str(err))
                    else:
                        answer = st.write_stream(stream)
                        _render_sources(sources, title="Literature sources")


# ═════════════════════════════════════════════════════════════════════════════
#  BATCH
# ═════════════════════════════════════════════════════════════════════════════
with tab_batch:
    batch_files = st.file_uploader(
        "Upload multiple retinal flatmount images",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    MAX_BATCH = 50

    if batch_files:
        if len(batch_files) > MAX_BATCH:
            st.warning(f"Maximum {MAX_BATCH} images per batch. Please remove some files.")
        st.markdown(
            f"""
        <p style="color:#9A8B6F;font-size:0.85rem;margin:0.3rem 0 1rem 0;">
            {len(batch_files)} image{"s" if len(batch_files) != 1 else ""} queued
        </p>
        """,
            unsafe_allow_html=True,
        )

        run_batch = st.button(
            "Run Batch Segmentation",
            type="primary",
            use_container_width=True,
            key="run_batch",
            disabled=len(batch_files) > MAX_BATCH,
        )

        if run_batch:
            progress_bar = st.progress(0, text="Starting…")
            results = []  # list of dicts for CSV

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, f in enumerate(batch_files):
                    stem = Path(f.name).stem
                    progress_bar.progress(idx / len(batch_files), text=f"Processing {f.name}…")

                    try:
                        img_pil = Image.open(f).convert("RGB")
                    except (Image.UnidentifiedImageError, OSError) as e:
                        st.warning(f"Skipping {f.name}: could not open image ({e})")
                        continue
                    img_np = np.array(img_pil)
                    h, w = img_np.shape[:2]

                    try:
                        masks_prob, _ = predict_single(
                            model,
                            img_np,
                            preprocess,
                            device,
                            config,
                            tta=tta,
                            threshold=0.5,
                        )
                    except Exception as e:
                        st.warning(f"Skipping {f.name}: segmentation failed ({e})")
                        continue
                    raw = [
                        (masks_prob[i] > thresholds[name]).astype(np.uint8)
                        for i, name in enumerate(config.mask_names)
                    ]
                    batch_stem = Path(f.name).stem
                    batch_vessel = load_vessel_mask(
                        batch_stem, config.manifest_path, config.data_root
                    )
                    masks_binary = postprocess_all(
                        np.stack(raw), config.mask_names, vessel_mask=batch_vessel, config=config
                    )

                    # Retinal area denominator (Connor et al. 2009 standard)
                    retinal_mask_b = np.zeros((h, w), dtype=np.uint8)
                    for _m in masks_binary:
                        retinal_mask_b |= _m
                    total_px = int(retinal_mask_b.sum()) or h * w

                    row = {"filename": f.name}
                    for i, name in enumerate(config.mask_names):
                        px = int(masks_binary[i].sum())
                        pct = round(100 * px / total_px, 4)
                        row[f"{name}_px"] = px
                        row[f"{name}_pct"] = pct
                        # Save binary mask PNG
                        mask_pil = Image.fromarray(masks_binary[i] * 255)
                        mask_buf = io.BytesIO()
                        mask_pil.save(mask_buf, format="PNG")
                        zf.writestr(f"masks/{stem}_{name}.png", mask_buf.getvalue())

                    results.append(row)
                    progress_bar.progress((idx + 1) / len(batch_files), text=f"Processed {f.name}")

                # Write CSV
                csv_buf = io.StringIO()
                fieldnames = (
                    ["filename"]
                    + [f"{n}_px" for n in config.mask_names]
                    + [f"{n}_pct" for n in config.mask_names]
                )
                writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
                zf.writestr("results.csv", csv_buf.getvalue())

            progress_bar.progress(1.0, text="Done.")

            # ── Summary table ─────────────────────────────────────────────
            section_header("Batch Results")
            mask_names = list(config.mask_names)
            header_cells = "".join(
                f'<th style="padding:0.55rem 1.2rem;text-align:left;color:#C5A55A;'
                f"font-family:'Playfair Display',serif;font-weight:600;font-size:0.85rem;"
                f'letter-spacing:0.06em;border-bottom:1px solid rgba(197,165,90,0.25);">'
                f"{MASK_LABELS.get(n, n.upper())} %</th>"
                for n in mask_names
            )
            rows_html = ""
            for ri, row in enumerate(results):
                bg = "rgba(197,165,90,0.04)" if ri % 2 == 0 else "transparent"
                fname_cell = (
                    f'<td style="padding:0.5rem 1.2rem;color:#BFB39A;font-size:0.82rem;'
                    f"border-bottom:1px solid rgba(197,165,90,0.08);white-space:nowrap;"
                    f'max-width:260px;overflow:hidden;text-overflow:ellipsis;">'
                    f"{html_escape(row['filename'])}</td>"
                )
                val_cells = "".join(
                    f'<td style="padding:0.5rem 1.2rem;color:#E8D5A3;font-size:0.82rem;'
                    f"font-family:'Inter',sans-serif;text-align:left;"
                    f'border-bottom:1px solid rgba(197,165,90,0.08);">'
                    f"{row[f'{n}_pct']:.2f}%</td>"
                    for n in mask_names
                )
                rows_html += f'<tr style="background:{bg};">{fname_cell}{val_cells}</tr>'

            st.markdown(
                f"""
            <div style="overflow-x:auto;border:1px solid rgba(197,165,90,0.22);
                        border-radius:12px;backdrop-filter:blur(16px);
                        background:rgba(20,18,14,0.62);margin-bottom:1rem;">
                <table style="width:100%;border-collapse:collapse;">
                    <thead>
                        <tr>
                            <th style="padding:0.55rem 1.2rem;text-align:left;color:#C5A55A;
                                font-family:'Playfair Display',serif;font-weight:600;font-size:0.85rem;
                                letter-spacing:0.06em;border-bottom:1px solid rgba(197,165,90,0.25);">
                                Filename</th>
                            {header_cells}
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # ── Download ZIP ───────────────────────────────────────────────
            section_header("Download")
            st.download_button(
                label=f"Download ZIP  ({len(batch_files)} images · masks + CSV)",
                data=zip_buf.getvalue(),
                file_name="oirseg_batch.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
#  LITERATURE
# ═════════════════════════════════════════════════════════════════════════════
with tab_lit:
    if not _rag_available():
        st.error("Could not import PubMed RAG module. Check that all dependencies are installed.")
    else:
        with st.expander("Ingest from PubMed"):
            pubmed_queries = st.text_area(
                "Search queries (one per line)",
                placeholder="retinopathy of prematurity\nVEGF retina\nneovascularization OIR",
                height=120,
                key="pubmed_query_input",
            )
            pubmed_max = st.number_input(
                "Max results per query",
                min_value=10,
                max_value=1000,
                value=200,
                step=10,
                key="pubmed_max_results",
            )
            if st.button(
                "Fetch from PubMed", key="ingest_pubmed_btn", disabled=not pubmed_queries.strip()
            ):
                queries = [q.strip() for q in pubmed_queries.strip().splitlines() if q.strip()]
                with st.spinner(f"Fetching {len(queries)} query/queries from PubMed…"):
                    from ingest import ingest as _ingest_pubmed

                    embed_model, _ = _get_rag_retriever()
                    n = _ingest_pubmed(queries, max_per_query=pubmed_max, model=embed_model)
                if n:
                    st.success(f"Added {n} new abstract(s) to the database.")
                else:
                    st.info(
                        "No new abstracts found — may already be ingested or no results matched."
                    )

        with st.expander("Ingest local data"):
            local_files = st.file_uploader(
                "Upload .txt, .pdf, or .csv (OIR dataset) files",
                type=["txt", "pdf", "csv"],
                accept_multiple_files=True,
                key="local_ingest_upload",
            )
            if st.button("Ingest files", key="ingest_local_btn", disabled=not local_files):
                import tempfile

                tmp_dir = Path(tempfile.mkdtemp())
                try:
                    tmp_paths = []
                    for uf in local_files:
                        safe_name = Path(uf.name).name
                        if not safe_name:
                            st.warning(f"Skipping file with invalid name: {uf.name}")
                            continue
                        p = tmp_dir / safe_name
                        p.write_bytes(uf.read())
                        tmp_paths.append(p)
                    with st.spinner(f"Ingesting {len(tmp_paths)} file(s)…"):
                        from ingest import ingest_local as _ingest_local

                        n = _ingest_local(tmp_paths)
                    if n:
                        st.success(f"Added {n} new chunk(s) to the database.")
                    else:
                        st.info("No new content found — files may already be ingested.")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        section_header("PubMed Literature Search")

        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        mode = st.radio(
            "Mode",
            ["Search abstracts", "Chat"],
            horizontal=True,
            help="Search returns ranked abstracts. Chat lets you have a multi-turn conversation grounded in PubMed.",
        )

        if mode == "Search abstracts":
            query = st.text_input(
                "Query",
                placeholder="e.g. VEGF retinopathy prematurity",
            )
            if st.button("Search", type="primary", use_container_width=True, key="run_search"):
                if query.strip():
                    with st.spinner("Searching PubMed index…"):
                        results = _ensure_rag().retrieve(query.strip())
                    if not results:
                        st.info(
                            "No results found. Make sure the PubMed RAG database has been ingested (`python ingest.py`)."
                        )
                    else:
                        section_header(f"{len(results)} abstracts found")
                        for src in results:
                            label = (
                                f"[{src['score']:.2f}]  {src['title']}  ({src['year']})"
                                if src["year"]
                                else f"[{src['score']:.2f}]  {src['title']}"
                            )
                            with st.expander(label):
                                if src["authors"]:
                                    st.markdown(f"**{src['authors']}** · *{src['journal']}*")
                                elif src["filename"]:
                                    st.caption(f"Local file: {src['filename']}")
                                st.write(src["text"])
                                if src["url"]:
                                    st.link_button("Open in PubMed", src["url"])

        else:  # Chat mode
            if st.session_state.rag_messages:
                if st.button("Clear chat", key="clear_chat"):
                    st.session_state.rag_messages = []
                    st.rerun()

            for msg in st.session_state.rag_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        _render_sources(msg["sources"])

            if prompt := st.chat_input("Ask about OIR / ROP / retinal vasculature…"):
                with st.chat_message("user"):
                    st.write(prompt)

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.rag_messages
                ]
                try:
                    sources, stream = _ensure_rag().chat_stream(prompt, history)
                except RuntimeError as err:
                    st.error(str(err))
                else:
                    with st.chat_message("assistant"):
                        answer = st.write_stream(stream)
                        _render_sources(sources)

                    st.session_state.rag_messages.append({"role": "user", "content": prompt})
                    st.session_state.rag_messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                    if len(st.session_state.rag_messages) > _MAX_RAG_MESSAGES:
                        st.session_state.rag_messages = st.session_state.rag_messages[
                            -_MAX_RAG_MESSAGES:
                        ]
