"""Tests for pure utility functions in ingest.py."""

from ingest import _chunk_text, _oir_csv_to_text

# ── _chunk_text ───────────────────────────────────────────────────────────────


def test_chunk_text_short_fits_in_one_chunk():
    text = "Short paragraph."
    chunks = _chunk_text(text, chunk_size=200)
    assert chunks == ["Short paragraph."]


def test_chunk_text_splits_on_paragraph_boundary():
    para1 = "A" * 400
    para2 = "B" * 400
    text = f"{para1}\n\n{para2}"
    chunks = _chunk_text(text, chunk_size=500)
    assert len(chunks) == 2
    assert chunks[0] == para1
    assert chunks[1] == para2


def test_chunk_text_long_paragraph_split_by_sentences():
    # Single paragraph longer than chunk_size, no double newlines
    sentences = ["Word. "] * 60  # ~360 chars
    text = "".join(sentences).strip()
    chunks = _chunk_text(text, chunk_size=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 120  # some slack for sentence boundaries


def test_chunk_text_empty_returns_empty():
    assert _chunk_text("", chunk_size=200) == []


def test_chunk_text_whitespace_only_returns_empty():
    assert _chunk_text("   \n\n   ", chunk_size=200) == []


def test_chunk_text_preserves_content():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = _chunk_text(text, chunk_size=50)
    combined = " ".join(chunks)
    assert "First paragraph" in combined
    assert "Second paragraph" in combined


# ── _oir_csv_to_text ──────────────────────────────────────────────────────────


def test_oir_csv_basic(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "Names,NOX_vs_OIR,Age_at_Sac,Treatment age,Treatment,Dose\nMouse1,OIR,17,12,PBS,\n",
        encoding="utf-8",
    )
    text = _oir_csv_to_text(csv_file)
    assert "Mouse1" in text
    assert "OIR" in text
    assert "P17" in text
    assert "PBS" in text


def test_oir_csv_skips_empty_names(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "Names,NOX_vs_OIR,Age_at_Sac,Treatment age,Treatment,Dose\n"
        ",OIR,17,,PBS,\n"
        "Mouse2,NOX,12,,,\n",
        encoding="utf-8",
    )
    text = _oir_csv_to_text(csv_file)
    assert "Mouse2" in text
    # Row with empty name should be skipped entirely
    assert text.count("Retinal flatmount") == 1


def test_oir_csv_includes_dose(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "Names,NOX_vs_OIR,Age_at_Sac,Treatment age,Treatment,Dose\nMouse3,OIR,17,12,DrugX,5mg/kg\n",
        encoding="utf-8",
    )
    text = _oir_csv_to_text(csv_file)
    assert "5mg/kg" in text
    assert "DrugX" in text


def test_oir_csv_empty_file(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text(
        "Names,NOX_vs_OIR,Age_at_Sac,Treatment age,Treatment,Dose\n",
        encoding="utf-8",
    )
    assert _oir_csv_to_text(csv_file) == ""
