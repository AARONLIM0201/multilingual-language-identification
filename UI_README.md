# Streamlit UI Prototype

This Streamlit app demonstrates the FYP language identification pipeline:

- Wav2Vec2 for uploaded speech audio
- OCR plus DistilBERT for uploaded text/handwriting images
- Late fusion when both audio and text predictions are available
- Saved project result dashboard using the existing JSON and PNG artifacts

## Run

Install the UI dependencies in your project environment, then run:

```powershell
pip install -r streamlit_requirements.txt
python -m streamlit run streamlit_app.py
```

Or use the helper script, which tries `fypenv`, Anaconda, then default Python:

```powershell
.\run_streamlit_ui.ps1
```

If PowerShell blocks the script, run this command in the project folder instead:

```powershell
python -m streamlit run streamlit_app.py
```

The app expects these local project files:

- `best_wav2vec2_model_7p5h/`
- `distilbert_grid_best.pth`
- `hf_cache/` for offline Hugging Face tokenizer/model files
- `wav2vec2_distilbert_full_analysis_results.json`
- generated result plots such as `01_method_comparison.png`

At the moment, this workspace cache contains `facebook/wav2vec2-base`. The
DistilBERT base model/tokenizer and TrOCR model may still need to be downloaded.
After installing dependencies, run this once while online:

```powershell
python prepare_ui_models.py
```

Then the Streamlit sidebar's "Use local Hugging Face cache only" option can be
enabled for offline demos.

## First Demo Scope

The current version supports live/upload demo flows:

1. Record live audio or upload audio to classify with Wav2Vec2.
2. Upload an image to extract OCR text, then classify with DistilBERT.
3. Upload both to compare audio-only, text-only, and fusion predictions.

For the most stable presentation, use live microphone recording for audio and
prepared image uploads for OCR/text.
