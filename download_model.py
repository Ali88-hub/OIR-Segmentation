"""Download the OIRseg model checkpoint from Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

CHECKPOINT_PATH = Path("Model V4 output/checkpoints/best_model.pth")

# ── Update these after uploading to Hugging Face ─────────────────────────────
HF_REPO_ID = "Ali88-hub/OIRseg"  # your HF repo
HF_FILENAME = "best_model.pth"  # filename inside the repo


def ensure_checkpoint() -> Path:
    """Download the checkpoint if it doesn't exist locally. Returns the path."""
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    print(f"Downloading {HF_FILENAME} from {HF_REPO_ID}…")
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=str(CHECKPOINT_PATH.parent),
    )
    # hf_hub_download may place the file directly or in a subfolder
    downloaded = Path(downloaded)
    if downloaded != CHECKPOINT_PATH and downloaded.exists():
        downloaded.rename(CHECKPOINT_PATH)

    print(f"Checkpoint saved to {CHECKPOINT_PATH}")
    return CHECKPOINT_PATH


if __name__ == "__main__":
    ensure_checkpoint()
