---
title: OIRseg
emoji: 🔬
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: "1.55.0"
app_file: app.py
pinned: false
---

# OIRseg

Retinal image segmentation tool for Oxygen-Induced Retinopathy (OIR) research. Segments retinal flatmount images into three zones: **neovascular (NV)**, **vaso-obliterated (VO)**, and **retinal tissue**, with quantitative area measurements.

## Features

- **Multi-class segmentation** — U-Net with EfficientNet-B4 encoder, trained on retinal flatmounts
- **Single & batch processing** — upload one image or a batch for bulk analysis
- **Test-time augmentation (TTA)** — optional multi-transform inference for improved accuracy
- **Per-class threshold tuning** — adjustable confidence thresholds for NV, VO, and retina masks
- **NV post-processing** — vessel suppression and VO boundary zone filtering to reduce false positives
- **Downloadable results** — masks, overlays, and CSV metrics as a ZIP
- **PubMed RAG** (optional) — AI-powered interpretation of results using retrieved PubMed literature and Claude

## Project Structure

```
.
├── app.py                 # Streamlit web interface
├── src/
│   ├── config.py          # Model & training hyperparameters
│   ├── model.py           # U-Net architecture (segmentation-models-pytorch)
│   └── predict.py         # Inference & post-processing pipeline
├── rag.py                 # PubMed RAG: retrieve abstracts + Claude Q&A
├── ingest.py              # Fetch & embed PubMed abstracts into ChromaDB
├── config.py              # RAG configuration (API keys, search params)
├── theme.py               # Streamlit UI theme
├── requirements.txt       # Pinned Python dependencies
└── .env.example           # Environment variable template
```

## Setup

### Requirements

- Python 3.11
- Model checkpoint (`best_model.pth`) — downloaded automatically from Hugging Face on first run

### Installation

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first launch, the model checkpoint (~227 MB) is downloaded automatically from Hugging Face Hub.

## PubMed RAG (Optional)

Enables AI-powered interpretation of segmentation results using PubMed literature.

1. Copy `.env.example` to `.env` and add your API keys:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   NCBI_EMAIL=your@email.com
   ```

2. Ingest PubMed abstracts:
   ```bash
   python ingest.py
   ```

3. Restart the app — the RAG tab will become available.

## Model

| Component | Value |
|-----------|-------|
| Architecture | U-Net |
| Encoder | EfficientNet-B4 (ImageNet pretrained) |
| Decoder attention | scSE |
| Input size | 768 x 768 |
| Output classes | 3 (NV, VO, Retina) |
| Loss | BCE + Focal Tversky (class-weighted) |

## License

This project was developed for academic research purposes.
