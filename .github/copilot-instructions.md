# FYP: Multimodal Language Identifier

## Project Overview
This is a Final Year Project implementing a **multimodal language identification system** for English, Malay, and Chinese using **audio (speech) and visual (OCR text) modalities**.

- **Audio Models**: CNN with mel-spectrogram features + Wav2Vec2 fine-tuning for 3-class language classification
- **Visual Pipeline**: EasyOCR + synthetic text image generation from Wikipedia content
- **Dataset**: 1-hour audio per language (16kHz WAV) + 5,000 OCR images per language
- **Main Workspace**: Single Jupyter notebook (`Multimodal Language Identifier.ipynb`) with full ML pipeline

## Core Architecture

### Audio Processing Pipeline
```
Raw Audio (WAV 16kHz) → Segment (3-10s) → Feature Extraction → Model Training
                                          ↓
                            1. Mel-spectrogram + 2D CNN
                            2. Raw waveform + Wav2Vec2 (transfer learning)
```

**Key Classes**:
- `AudioSegmentDataset`: Segments audio files into fixed chunks with on-the-fly mel-spectrogram conversion
- `MelSpectrogramCNN`: 4-layer 2D CNN with BatchNorm + Dropout (256 → 128 → 3 classes)
- `Wav2Vec2Dataset`: Raw audio processing for transformer-based model (longer 10s segments)

**Data Split Strategy** (CRITICAL):
- Uses **temporal/sequential split** (not random) to prevent data leakage
- Train 70% / Val 15% / Test 15% split maintains temporal order within each file
- Files in `Dataset/audio_split/` folders are already split correctly

### Visual/OCR Pipeline
```
Wikipedia Crawling → Text Extraction → Image Generation → OCR Training Data
```

**Utilities**:
- `generate_image.py`: Creates synthetic text images with proper fonts (Noto Sans for EN/MY, Noto Sans SC for Chinese)
- `crawl_wikipedia()`: Multi-article crawler with simplified Chinese conversion (using OpenCC `t2s`)
- Chinese-specific: Filters ASCII characters, ensures Simplified Chinese with `opencc` library
- Output: `ocr_images/{EN,MY,CN}/` with corresponding `labels.csv`

## Critical Workflows

### Environment Setup
```powershell
# Activate virtual environment (ALWAYS required)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\fypenv\Scripts\Activate.ps1

# CUDA-enabled PyTorch required for GPU training (20-60x speedup for wav2vec2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Preparation
1. **Audio compilation**: Uses FFmpeg to concatenate clips into 1-hour WAV files (16kHz, 16-bit PCM)
   - Source: `Dataset/{en,ms,zh}/{en,ms,zh}/` (raw clips)
   - Output: `Dataset/compiled_audio/{en,ms,zh}_1hour.wav`
   - Split: `Dataset/audio_active/` (60%) + `Dataset/audio_reserved/` (40%)

2. **OCR image generation**: Run cells in notebook or `generate_image.py`
   - Requires: Wikipedia URL seeds in `CATEGORY_URLS` dict
   - Chinese processing: Traditional → Simplified + ASCII filtering enabled by default

### Model Training
**Order of operations in notebook**:
1. Check GPU availability (Cell #17-18) - CRITICAL for wav2vec2
2. Load and split audio data with temporal ordering (Cell #19-22)
3. Train CNN mel-spectrogram model (Cell #23-28) - ~30 epochs baseline
4. Train Wav2Vec2 model (Cell #39-42) - Only classification head trainable, encoder frozen

**Model Checkpoints** (saved in root):
- `best_cnn_fixed_model.pth` - CNN with temporal split (current best practice)
- `best_cnn_melspec_model.pth` - Original CNN (has data leakage, deprecated)
- `best_cnn_melspec_weighted_model.pth` - With class weighting
- `best_wav2vec2_model.pth` - Fine-tuned transformer model

## Project-Specific Conventions

### Data Leakage Prevention
**Background**: Initial models showed 100% accuracy (wav2vec2) or 97% EN/MS with 0% ZH recall (CNN) due to random splitting of audio segments from same file.

**Solution**: Temporal/sequential split implemented in cells around line 630-690
```python
# CORRECT: Split by time windows, not random shuffle
train_cutoff = int(TRAIN_RATIO * len(segments))
val_cutoff = train_cutoff + int(VAL_RATIO * len(segments))
```

**Rule**: Never use `random.shuffle()` on audio segments - maintain temporal order per language file.

### Language Code Mapping
```python
LANG_MAP = {'en': 0, 'ms': 1, 'zh': 2}
LANG_NAMES = ['English', 'Malay', 'Chinese']
```
- EN = English, MY/MS = Malay/Bahasa Malaysia, CN/ZH = Chinese (Simplified Mandarin)
- File prefixes use 2-letter codes: `en_`, `ms_`, `zh_`

### Audio Configuration Constants
```python
SAMPLE_RATE = 16000       # Fixed for wav2vec2 compatibility
SEGMENT_LENGTH = 3        # CNN: 3s segments
SEGMENT_LENGTH = 10       # Wav2Vec2: 10s segments (longer context)
N_MELS = 128             # Mel-spectrogram bins
BATCH_SIZE = 32          # CNN training
BATCH_SIZE = 8           # Wav2Vec2 (GPU memory constraint)
```

### FFmpeg Usage
- FFmpeg binaries located in `ffmpeg/ffmpeg-2026-01-07-git-af6a1dd0b2-essentials_build/bin/`
- Used for: Audio concatenation, format conversion, duration probing
- Check availability: `has_ffmpeg()` function in notebook cell #10

## Key Dependencies
- **PyTorch** (CUDA-enabled): Deep learning framework
- **librosa**: Audio loading and mel-spectrogram extraction
- **transformers** (Hugging Face): Wav2Vec2 models
- **EasyOCR**: OCR for text images (cv2, PIL backends)
- **FFmpeg**: Audio processing (external binary)
- **OpenCC**: Traditional→Simplified Chinese conversion
- **BeautifulSoup**: Wikipedia web scraping

## Common Pitfalls

1. **Missing GPU**: Wav2Vec2 training is 20-60x slower on CPU. Always verify `torch.cuda.is_available()` before training.

2. **Audio file paths**: Windows paths require raw strings `r"C:\Users\..."` or escaped backslashes.

3. **Chinese text filtering**: `CN_FILTER_ASCII = True` removes English mixed in Chinese text. Disable if multilingual Chinese text is expected.

4. **Segment length mismatch**: CNN uses 3s segments, Wav2Vec2 uses 10s. Don't mix datasets between models.

5. **Model device mismatch**: Always move model and data to same device:
   ```python
   model = model.to(device)
   inputs = inputs.to(device)
   ```

## File Navigation
- **Main notebook**: [Multimodal Language Identifier.ipynb](Multimodal Language Identifier.ipynb)
- **Data**: `Dataset/` (audio), `ocr_images/` (visual)
- **Models**: `.pth` files in root directory
- **Utilities**: [generate_image.py](generate_image.py), [Activation.txt](Activation.txt) (venv setup)
- **Fonts**: `Language Font/Noto_Sans/` (EN/MY), `Language Font/Noto_Sans_SC/` (CN)
