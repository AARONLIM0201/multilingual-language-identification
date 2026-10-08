# FYP: Multimodal Language Identifier

This repository contains my Final Year Project on multimodal language identification for **English, Malay, and Mandarin**.

The project combines multiple signals:

- **Audio classification** with Wav2Vec2 and CNN-based models
- **OCR / text classification** with DistilBERT and OCR-generated text samples
- **Late fusion** with equal weighting, accuracy-weighted fusion, and confidence-adaptive fusion

## Abstract

This project investigates multimodal language identification using speech and OCR-derived text cues. Audio-based recognition is explored with Wav2Vec2 and CNN models, while text recognition is handled with DistilBERT over OCR-generated samples. The final system compares several fusion strategies for English, Malay, and Mandarin, showing that multimodal fusion improves performance over single-modality baselines, with equal-weight fusion giving the strongest overall result.

## Project Goals

- Compare audio-only, text-only, and fused language identification models
- Evaluate whether fusion improves accuracy over single-modality baselines
- Produce reproducible figures, tables, and summary metrics for the report

## Repository Layout

- `*.ipynb` - notebooks for training, evaluation, and analysis
- `generate_image.py` - helper script for synthetic OCR image generation
- `mcnemar_analysis.py` - paired significance testing for fusion comparisons
- `*_metrics_summary.json` - saved evaluation summaries
- `*_test_predictions.csv` and `wav2vec2_distilbert_full_analysis_results.json` - prediction outputs used for analysis
- `figures/` and PNG files - plots used in the report and poster
- `README.md` - this overview and setup guide

## Main Results Artifacts

The following files are especially useful for reviewing the project results:

- `wav2vec2_distilbert_full_analysis_results.json`
- `mcnemar_results.json`
- `cnn_test_predictions.csv`
- `distilbert_test_predictions.csv`
- `wav2vec2_test_predictions.csv`
- `wav2vec2_final_test_results_7p5h.csv`
- `wav2vec2_final_metrics_7p5h.json`

## Results Snapshot

| Model | Accuracy |
| --- | ---: |
| Equal Weighting (0.5/0.5) | 99.18% |
| Confidence-Adaptive Fusion | 99.01% |
| Accuracy-Weighted Fusion | 97.86% |
| Text Only (DistilBERT) | 97.28% |
| Audio Only (Wav2Vec2) | 94.16% |

The paired McNemar analysis in `mcnemar_results.json` shows that Equal Weighting is significantly better than Audio Only, Text Only, and Accuracy-Weighted Fusion, while the difference versus Confidence-Adaptive Fusion is not statistically significant.

## Reproducibility Notes

- The repository keeps the analysis outputs and figures needed to understand the experiments.
- Large raw data folders are intentionally excluded so the GitHub repository stays manageable.
- Model checkpoints are not committed by default. If you want to share them, use Git LFS or an external release.

## Setup

Activate the Windows virtual environment first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\fypenv\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
python -m pip install -r streamlit_requirements.txt
```

## How to Use

1. Open the notebooks in VS Code or Jupyter.
2. Run the cells for the audio, OCR, or fusion pipeline you want to inspect.
3. Use the saved JSON and CSV outputs for reporting, plotting, or statistical testing.

For the significance analysis, run:

```powershell
python mcnemar_analysis.py -i wav2vec2_distilbert_full_analysis_results.json -o mcnemar_results.json
```

## How to Reproduce

1. Clone the repository.
2. Activate the Windows virtual environment.
3. Install the required Python packages.
4. Run the notebooks in this general order:
	- OCR and text generation notebooks
	- audio model notebooks
	- fusion notebook
	- significance analysis with `mcnemar_analysis.py`
5. Review the generated CSV and JSON outputs for the final tables and figures.

Suggested notebook order:

1. `Multimodal Language Identifier.ipynb`
2. `OCR_Image_Noise.ipynb`
3. `Wav2Vec2+DistilBERT.ipynb`
4. `7.5h wav2vec2 (audio Transformer).ipynb`
5. `DistilBERT.ipynb`

## GitHub Upload

This repository has not been pushed to GitHub yet. There is currently no configured remote, so the next step is to create a GitHub repository and connect it locally.

Example commands:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git add .
git commit -m "Prepare FYP repository for GitHub"
git push -u origin main
```

## Notes

- If you want to include trained checkpoints, use Git LFS rather than regular Git.
- If you want a fully reproducible public repo, add a `requirements.txt` or `environment.yml` that matches the runtime used for the final experiments.
