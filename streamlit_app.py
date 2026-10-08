import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
LANGUAGE_NAMES = ["English", "Malay", "Mandarin"]
LANGUAGE_IDS = {"en": 0, "ms": 1, "zh": 2}
SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 10
AUDIO_CHUNK_OVERLAP_SECONDS = 2
AUDIO_MAX_CHUNKS = 8
AUDIO_MIN_RECOMMENDED_SECONDS = 3.0

AUDIO_MODEL_PATH = ROOT / "best_wav2vec2_model_7p5h"
TEXT_MODEL_PATH = ROOT / "distilbert_grid_best.pth"
TEXT_MODEL_NAME = "distilbert-base-multilingual-cased"
HF_CACHE_DIR = ROOT / "hf_cache"
RESULTS_PATH = ROOT / "wav2vec2_distilbert_full_analysis_results.json"

OCR_COMPARISON = pd.DataFrame(
    {
        "Language": ["English", "Malay", "Mandarin"],
        "TrOCR Accuracy (%)": [62.24, 46.65, 2.56],
        "EasyOCR Accuracy (%)": [87.90, 87.62, 76.82],
        "Selected Engine": ["EasyOCR", "EasyOCR", "EasyOCR"],
    }
)


st.set_page_config(
    page_title="Wav2Vec2 + DistilBERT Language Identifier",
    layout="wide",
)


def softmax_label_table(probs):
    data = pd.DataFrame(
        {
            "Language": LANGUAGE_NAMES,
            "Confidence": [float(x) for x in probs],
        }
    )
    data["Confidence (%)"] = data["Confidence"] * 100
    return data


def best_prediction(probs):
    idx = int(np.argmax(probs))
    return LANGUAGE_NAMES[idx], float(probs[idx])


def fuse_equal(audio_probs, text_probs):
    return 0.5 * audio_probs + 0.5 * text_probs


def fuse_accuracy_weighted(audio_probs, text_probs, w_audio=0.6, w_text=0.4):
    return w_audio * audio_probs + w_text * text_probs


def fuse_confidence_adaptive(audio_probs, text_probs, threshold=0.8):
    audio_conf = float(np.max(audio_probs))
    text_conf = float(np.max(text_probs))

    if text_conf > threshold:
        return 0.45 * audio_probs + 0.55 * text_probs
    if audio_conf > threshold:
        return 0.65 * audio_probs + 0.35 * text_probs
    return fuse_equal(audio_probs, text_probs)


def render_prediction(title, probs):
    label, confidence = best_prediction(probs)
    st.metric(title, label, f"{confidence * 100:.2f}% confidence")
    table = softmax_label_table(probs)
    st.bar_chart(table.set_index("Language")["Confidence (%)"])
    st.dataframe(
        table[["Language", "Confidence (%)"]],
        hide_index=True,
        use_container_width=True,
    )


@st.cache_resource(show_spinner=False)
def load_audio_model(local_files_only=True):
    import torch
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base",
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=local_files_only,
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        str(AUDIO_MODEL_PATH),
        num_labels=len(LANGUAGE_NAMES),
        local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


@st.cache_resource(show_spinner=False)
def load_text_model(local_files_only=True):
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained(
        TEXT_MODEL_NAME,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=local_files_only,
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        TEXT_MODEL_NAME,
        num_labels=len(LANGUAGE_NAMES),
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=local_files_only,
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(str(TEXT_MODEL_PATH), map_location="cpu")
    model.load_state_dict(state_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_easyocr_reader(expected_language):
    import easyocr
    import torch

    gpu = torch.cuda.is_available()
    if expected_language == "Malay":
        return easyocr.Reader(["ms", "en"], gpu=gpu)
    if expected_language == "Mandarin":
        return easyocr.Reader(["ch_sim", "en"], gpu=gpu)
    return easyocr.Reader(["en"], gpu=gpu)


@st.cache_resource(show_spinner=False)
def load_trocr(local_files_only=True):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-printed",
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=local_files_only,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed",
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=local_files_only,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def normalize_waveform(waveform):
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.nan_to_num(waveform)
    if waveform.size == 0:
        return waveform

    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform / peak * 0.95
    return waveform


def make_audio_chunks(
    waveform,
    chunk_seconds=AUDIO_CHUNK_SECONDS,
    overlap_seconds=AUDIO_CHUNK_OVERLAP_SECONDS,
    max_chunks=AUDIO_MAX_CHUNKS,
):
    chunk_len = int(chunk_seconds * SAMPLE_RATE)
    hop_len = max(1, int((chunk_seconds - overlap_seconds) * SAMPLE_RATE))

    if waveform.size <= chunk_len:
        return [waveform]

    starts = list(range(0, waveform.size - chunk_len + 1, hop_len))
    tail_start = waveform.size - chunk_len
    if starts[-1] != tail_start:
        starts.append(tail_start)

    if len(starts) > max_chunks:
        sampled = np.linspace(0, len(starts) - 1, max_chunks).round().astype(int)
        starts = [starts[i] for i in sorted(set(sampled))]

    return [waveform[start : start + chunk_len] for start in starts]


def predict_audio(uploaded_file, local_files_only=True, robust_audio=True, return_details=False):
    import librosa
    import torch
    import torch.nn.functional as F

    processor, model, device = load_audio_model(local_files_only=local_files_only)
    suffix = Path(getattr(uploaded_file, "name", "audio.wav")).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        waveform, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)

    original_duration = float(len(waveform) / SAMPLE_RATE) if len(waveform) else 0.0
    waveform = normalize_waveform(waveform)

    if robust_audio and waveform.size:
        trimmed, _ = librosa.effects.trim(waveform, top_db=30)
        if trimmed.size >= int(0.75 * SAMPLE_RATE):
            waveform = trimmed

    processed_duration = float(len(waveform) / SAMPLE_RATE) if len(waveform) else 0.0
    chunks = make_audio_chunks(waveform) if robust_audio else [waveform]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]
    if not chunks:
        raise ValueError("No usable audio was found after loading the file.")

    probs_batches = []
    with torch.no_grad():
        for start in range(0, len(chunks), 4):
            batch = chunks[start : start + 4]
            inputs = processor(
                batch,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs_batches.append(F.softmax(outputs.logits, dim=-1).cpu().numpy())

    chunk_probs = np.vstack(probs_batches)
    probs = chunk_probs.mean(axis=0)
    probs = probs / probs.sum()

    if not return_details:
        return probs

    chunk_rows = []
    for idx, chunk_prob in enumerate(chunk_probs, start=1):
        label, confidence = best_prediction(chunk_prob)
        chunk_rows.append(
            {
                "Chunk": idx,
                "Prediction": label,
                "Confidence (%)": round(confidence * 100, 2),
            }
        )

    details = {
        "original_duration": original_duration,
        "processed_duration": processed_duration,
        "chunk_count": len(chunks),
        "chunk_rows": chunk_rows,
    }
    return probs, details


def prepare_ocr_image_file(uploaded_file, preprocess=False):
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    image = Image.open(uploaded_file).convert("RGB")
    if preprocess:
        image = ImageOps.grayscale(image)
        width, height = image.size
        image = image.resize((width * 2, height * 2))
        image = ImageEnhance.Contrast(image).enhance(1.8)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.point(lambda pixel: 255 if pixel > 165 else 0)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(tmp.name)
    tmp.close()
    return tmp.name


def extract_text_easyocr(uploaded_file, expected_language="Unknown / mixed", preprocess=False):
    reader = load_easyocr_reader(expected_language)
    tmp_path = prepare_ocr_image_file(uploaded_file, preprocess=preprocess)

    try:
        result = reader.readtext(
            tmp_path,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            mag_ratio=2.0,
        )
    finally:
        os.unlink(tmp_path)

    if not isinstance(result, list):
        return str(result).strip(), []

    text_parts = []
    detail_rows = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            text = str(item[1]).strip()
            confidence = float(item[2])
            if text:
                text_parts.append(text)
                detail_rows.append({"Text": text, "Confidence": confidence})

    return " ".join(text_parts).strip(), detail_rows


def ocr_quality_score(text, details, candidate_language):
    if not text:
        return 0.0

    confidence_values = [
        float(row.get("Confidence", 0.0))
        for row in details
        if isinstance(row, dict)
    ]
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.50
    chinese_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    alnum_chars = sum(ch.isalnum() for ch in text)
    odd_symbol_chars = sum(
        not (ch.isalnum() or ch.isspace() or "\u4e00" <= ch <= "\u9fff" or ch in ".,;:!?'-\"()[]")
        for ch in text
    )
    quality = alnum_chars / max(len(text), 1)
    length_bonus = min(len(text) / 40.0, 1.0)
    odd_symbol_penalty = odd_symbol_chars / max(len(text), 1)

    score = (avg_confidence * 0.55) + (quality * 0.30) + (length_bonus * 0.15)
    if candidate_language == "Mandarin" and chinese_chars > 0:
        score += 0.45
    if candidate_language != "Mandarin" and chinese_chars > 0:
        score -= 0.20
    score -= odd_symbol_penalty * 0.35
    return score


def extract_text_auto_ocr(uploaded_file, preprocess=False):
    candidates = []
    for candidate_language in ["Mandarin", "Malay", "English"]:
        text, details = extract_text_easyocr(
            uploaded_file,
            expected_language=candidate_language,
            preprocess=preprocess,
        )
        candidates.append(
            {
                "candidate_language": candidate_language,
                "engine": "EasyOCR",
                "text": text,
                "details": details,
                "score": ocr_quality_score(text, details, candidate_language),
            }
        )

    best = max(candidates, key=lambda row: row["score"])
    summary_rows = [
        {
            "Candidate": row["candidate_language"],
            "Engine": row["engine"],
            "Score": round(row["score"], 4),
            "Extracted Text": row["text"],
        }
        for row in candidates
    ]
    return best["text"], f"Auto OCR -> {best['engine']} ({best['candidate_language']})", summary_rows


def extract_text_trocr(uploaded_file, local_files_only=True, preprocess=False):
    import torch
    from PIL import Image

    processor, model, device = load_trocr(local_files_only=local_files_only)
    if preprocess:
        tmp_path = prepare_ocr_image_file(uploaded_file, preprocess=True)
        try:
            image = Image.open(tmp_path).convert("RGB")
        finally:
            os.unlink(tmp_path)
    else:
        image = Image.open(uploaded_file).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=96, num_beams=5)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip(), []


def _rewind_upload(uploaded_file):
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)


def hybrid_candidate_languages(expected_language):
    if expected_language in {"English", "Malay", "Mandarin"}:
        return [expected_language]
    return ["Mandarin", "Malay", "English"]


def extract_text_hybrid_ocr(
    uploaded_file,
    expected_language="Unknown / mixed",
    local_files_only=True,
    preprocess=False,
):
    candidates = []
    candidate_languages = hybrid_candidate_languages(expected_language)

    for candidate_language in candidate_languages:
        _rewind_upload(uploaded_file)
        text, details = extract_text_easyocr(
            uploaded_file,
            expected_language=candidate_language,
            preprocess=preprocess,
        )
        candidates.append(
            {
                "candidate_language": candidate_language,
                "engine": "EasyOCR",
                "text": text,
                "details": details,
                "score": ocr_quality_score(text, details, candidate_language),
            }
        )

    _rewind_upload(uploaded_file)
    trocr_text, trocr_details = extract_text_trocr(
        uploaded_file,
        local_files_only=local_files_only,
        preprocess=preprocess,
    )
    trocr_language = expected_language if expected_language in {"English", "Malay", "Mandarin"} else "English"
    candidates.append(
        {
            "candidate_language": trocr_language,
            "engine": "TrOCR",
            "text": trocr_text,
            "details": trocr_details,
            "score": ocr_quality_score(trocr_text, trocr_details, trocr_language),
        }
    )

    best = max(candidates, key=lambda row: row["score"])
    summary_rows = [
        {
            "Candidate": row["candidate_language"],
            "Engine": row["engine"],
            "Score": round(row["score"], 4),
            "Extracted Text": row["text"],
        }
        for row in sorted(candidates, key=lambda row: row["score"], reverse=True)
    ]
    engine_label = f"Hybrid OCR -> {best['engine']} ({best['candidate_language']})"
    return best["text"], engine_label, summary_rows


def extract_text_with_strategy(
    uploaded_file,
    strategy,
    expected_language,
    local_files_only=True,
    preprocess=False,
):
    selected_engine = strategy
    if strategy == "Hybrid OCR":
        text, selected_engine, details = extract_text_hybrid_ocr(
            uploaded_file,
            expected_language=expected_language,
            local_files_only=local_files_only,
            preprocess=preprocess,
        )
        return text, selected_engine, details

    if selected_engine == "Auto OCR":
        text, selected_engine, details = extract_text_auto_ocr(
            uploaded_file,
            preprocess=preprocess,
        )
    elif selected_engine == "TrOCR":
        text, details = extract_text_trocr(
            uploaded_file,
            local_files_only=local_files_only,
            preprocess=preprocess,
        )
    else:
        text, details = extract_text_easyocr(
            uploaded_file,
            expected_language=expected_language,
            preprocess=preprocess,
        )

    return text, selected_engine, details


def predict_text(text, local_files_only=True):
    import torch
    import torch.nn.functional as F

    tokenizer, model, device = load_text_model(local_files_only=local_files_only)
    encodings = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    return probs


def load_results():
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_hf_model_cached(model_name):
    cache_folder = "models--" + model_name.replace("/", "--")
    return (HF_CACHE_DIR / cache_folder).exists()


def render_dashboard():
    results = load_results()
    if not results:
        st.info("No saved fusion result JSON was found yet.")
        return

    summary = results.get("summary_results", {})
    best = results.get("best_method", {})
    st.subheader("Saved Evaluation Results")

    cols = st.columns(5)
    for col, (name, acc) in zip(cols, summary.items()):
        col.metric(name, f"{float(acc) * 100:.2f}%")

    if best:
        st.success(
            f"Best saved method: {best.get('name')} "
            f"({float(best.get('accuracy', 0)) * 100:.2f}%)"
        )

    st.bar_chart(
        pd.DataFrame(
            {
                "Method": list(summary.keys()),
                "Accuracy (%)": [float(v) * 100 for v in summary.values()],
            }
        ).set_index("Method")
    )

    st.subheader("Generated Analysis Figures")
    figure_files = [
        "01_method_comparison.png",
        "02_per_language_comparison.png",
        "03_confidence_correlation.png",
        "04_modality_agreement.png",
        "05_confusion_matrices_comparison.png",
        "06_fusion_improvement.png",
    ]
    existing_figures = [ROOT / name for name in figure_files if (ROOT / name).exists()]
    for left, right in zip(existing_figures[::2], existing_figures[1::2]):
        c1, c2 = st.columns(2)
        c1.image(str(left), caption=left.name, use_container_width=True)
        c2.image(str(right), caption=right.name, use_container_width=True)
    if len(existing_figures) % 2 == 1:
        st.image(str(existing_figures[-1]), caption=existing_figures[-1].name)


def get_sample_texts():
    return {
        "English": "THIS KIND OF TRICK QUESTION IS OFTEN USED IN APTITUDE TESTS",
        "Malay": "SAYA SEDANG MENJALANKAN SISTEM PENGENALAN BAHASA UNTUK PROJEK AKHIR",
        "Mandarin": "这是一个用于语言识别的多模态系统",
    }


def render_project_overview():
    results = load_results()
    best = results.get("best_method", {}) if results else {}

    st.subheader("Project Demonstration Flow")
    cols = st.columns(3)
    cols[0].metric("Audio Model", "Wav2Vec2", "speech waveform")
    cols[1].metric("Text Model", "DistilBERT", "OCR text")
    if best:
        cols[2].metric(
            "Best Fusion",
            f"{float(best.get('accuracy', 0)) * 100:.2f}%",
            best.get("name", "late fusion"),
        )
    else:
        cols[2].metric("Fusion", "Late Fusion", "probability averaging")

    st.markdown(
        """
        The demo separates the system into the same stages used in the experiment:
        audio classification, OCR/text classification, and probability-level fusion.
        During presentation, run one modality first, then run both together to show
        why the fusion result is the main contribution.
        """
    )

    st.subheader("Suggested Presentation Script")
    st.write("1. Open **Project Results** and show that fusion outperforms the single-modality baselines.")
    st.write("2. Upload or enter text in **Live Demo** to show the DistilBERT branch.")
    st.write("3. Upload audio in **Live Demo** to show the Wav2Vec2 branch.")
    st.write("4. Provide both inputs and compare audio-only, text-only, and fused confidence scores.")

    st.subheader("OCR Engine Selection")
    st.dataframe(OCR_COMPARISON, hide_index=True, use_container_width=True)
    st.caption(
        "Hybrid OCR runs multiple OCR candidates and chooses the strongest "
        "extraction by confidence and text cleanliness."
    )


def show_model_error(area, error):
    area.error(
        "This step could not run. Check that the required package is installed, "
        "the checkpoint exists, and the model/tokenizer is available in hf_cache."
    )
    area.code(str(error))


def looks_like_noisy_ocr(text):
    if not text:
        return True
    useful_chars = sum(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text)
    return useful_chars / max(len(text), 1) < 0.55


def init_demo_state():
    defaults = {
        "audio_input_reset": 0,
        "image_input_reset": 0,
        "manual_text_value": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_audio_inputs():
    st.session_state.audio_input_reset += 1


def reset_image_input():
    st.session_state.image_input_reset += 1


def reset_manual_text():
    st.session_state.manual_text_value = ""


def reset_all_inputs():
    reset_audio_inputs()
    reset_image_input()
    reset_manual_text()


init_demo_state()

st.title("Wav2Vec2 + DistilBERT Language Identifier")
st.caption("Audio, OCR text, and late-fusion demo for English, Malay, and Mandarin.")

missing_cached_models = [
    name
    for name in [TEXT_MODEL_NAME, "microsoft/trocr-base-printed"]
    if not is_hf_model_cached(name)
]
if missing_cached_models:
    st.warning(
        "Some Hugging Face models are not cached yet: "
        + ", ".join(missing_cached_models)
        + ". Keep local-cache-only off for online download, or run prepare_ui_models.py once."
    )

with st.sidebar:
    st.header("Demo Settings")
    local_files_only = st.toggle(
        "Use local Hugging Face cache only",
        value=False,
        help="Turn this on for an offline demo after all required models have been downloaded once.",
    )
    ocr_engine = st.selectbox(
        "OCR strategy",
        ["Hybrid OCR", "EasyOCR", "TrOCR"],
        help="Hybrid OCR runs multiple OCR candidates and chooses the strongest extraction.",
    )
    expected_ocr_language = st.selectbox(
        "Expected image language",
        ["Unknown / mixed", "English", "Malay", "Mandarin"],
        help="Hybrid OCR needs this hint because OCR happens before language classification.",
    )
    preprocess_ocr = st.toggle(
        "Enhance image before OCR",
        value=True,
        help="Upscales, sharpens, increases contrast, and binarizes the image before OCR.",
    )
    fusion_method = st.selectbox(
        "Fusion method",
        ["Equal Weighting (0.5/0.5)", "Accuracy-Weighted Fusion", "Confidence-Adaptive Fusion"],
    )

    st.divider()
    st.caption("Cache status")
    st.write(f"Wav2Vec2 base: {'found' if is_hf_model_cached('facebook/wav2vec2-base') else 'missing'}")
    st.write(
        "DistilBERT base: "
        f"{'found' if is_hf_model_cached(TEXT_MODEL_NAME) else 'missing'}"
    )
    st.write(
        "TrOCR printed: "
        f"{'found' if is_hf_model_cached('microsoft/trocr-base-printed') else 'missing'}"
    )


tabs = st.tabs(["Live Demo", "Project Results", "Presentation", "Model Files"])

with tabs[0]:
    st.subheader("Run A Prediction")
    if ocr_engine == "Hybrid OCR":
        if expected_ocr_language == "Unknown / mixed":
            st.info(
                "Hybrid OCR is set to unknown/mixed, so the app will try TrOCR plus "
                "Mandarin, Malay, and English EasyOCR candidates, then choose the "
                "strongest extraction. This is slower than selecting one OCR engine."
            )
        else:
            st.info(
                f"Hybrid OCR will compare TrOCR with EasyOCR tuned for {expected_ocr_language}, "
                "then choose the strongest extraction."
            )

    audio_col, image_col = st.columns(2)

    with audio_col:
        st.markdown("**Audio Input**")
        recorded_audio = None
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input(
                "Record speech live",
                help="Record around 5-10 seconds for a cleaner Wav2Vec2 prediction.",
                key=f"recorded_audio_{st.session_state.audio_input_reset}",
            )
        else:
            st.info(
                "Your Streamlit version does not support browser audio recording yet. "
                "Use audio upload, or upgrade Streamlit."
            )

        uploaded_audio = st.file_uploader(
            "Upload speech audio",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            key=f"uploaded_audio_{st.session_state.audio_input_reset}",
        )
        audio_file = recorded_audio or uploaded_audio
        if audio_file is not None:
            st.audio(audio_file)
            st.caption("Audio source: live recording" if recorded_audio is not None else "Audio source: uploaded file")
        st.button(
            "Clear audio",
            use_container_width=True,
            on_click=reset_audio_inputs,
            help="Remove the current recording/upload so you can run a text-only demo or choose another audio file.",
        )

    with image_col:
        st.markdown("**Image/Text Input**")
        image_file = st.file_uploader(
            "Upload text image",
            type=["png", "jpg", "jpeg", "bmp"],
            key=f"image_file_{st.session_state.image_input_reset}",
        )
        if image_file is not None:
            st.image(image_file, use_container_width=True)
        st.button(
            "Clear image",
            use_container_width=True,
            on_click=reset_image_input,
            help="Remove the current image so you can run an audio-only demo or choose another image.",
        )

    sample_choice = st.selectbox(
        "Optional sample text",
        ["None"] + list(get_sample_texts().keys()),
        help="Useful for quickly demonstrating the DistilBERT branch without OCR.",
    )
    if sample_choice != "None" and st.button("Load sample text", use_container_width=True):
        st.session_state.manual_text_value = get_sample_texts()[sample_choice]

    manual_text = st.text_area(
        "OCR text / manual correction",
        key="manual_text_value",
        help="After OCR runs, you can correct the extracted text here before DistilBERT classification.",
    )

    clear_col, reset_col = st.columns(2)
    clear_col.button(
        "Clear OCR/manual text",
        use_container_width=True,
        on_click=reset_manual_text,
        help="Remove typed or OCR text so the next run can extract fresh text from an uploaded image.",
    )
    reset_col.button(
        "Clear all inputs",
        use_container_width=True,
        on_click=reset_all_inputs,
        help="Reset audio, image, and text inputs for the next demo scenario.",
    )

    run_button = st.button("Run Detection", type="primary", use_container_width=True)

    if run_button:
        audio_probs = None
        text_probs = None
        extracted_text = manual_text.strip()

        audio_result, text_result = st.columns(2)

        if audio_file is not None:
            with audio_result:
                with st.spinner("Running Wav2Vec2 audio model..."):
                    try:
                        audio_probs, audio_details = predict_audio(
                            audio_file,
                            local_files_only=local_files_only,
                            robust_audio=True,
                            return_details=True,
                        )
                        render_prediction("Wav2Vec2 audio prediction", audio_probs)
                        st.caption(
                            "Audio analysed: "
                            f"{audio_details['processed_duration']:.1f}s "
                            f"from {audio_details['original_duration']:.1f}s input | "
                            f"{audio_details['chunk_count']} chunk(s)"
                        )
                        if audio_details["processed_duration"] < AUDIO_MIN_RECOMMENDED_SECONDS:
                            st.warning(
                                "The usable speech is very short. For a stronger audio-only demo, "
                                "record around 5-10 seconds with minimal background noise."
                            )
                        if audio_details["chunk_count"] > 1:
                            with st.expander("Audio chunk predictions"):
                                st.dataframe(
                                    pd.DataFrame(audio_details["chunk_rows"]),
                                    hide_index=True,
                                    use_container_width=True,
                                )
                    except Exception as exc:
                        show_model_error(st, exc)
        else:
            audio_result.info("Record or upload audio to run Wav2Vec2.")

        if image_file is not None and not extracted_text:
            with text_result:
                with st.spinner(f"Extracting text with {ocr_engine}..."):
                    try:
                        extracted_text, selected_engine, details = extract_text_with_strategy(
                            image_file,
                            ocr_engine,
                            expected_ocr_language,
                            local_files_only=local_files_only,
                            preprocess=preprocess_ocr,
                        )
                        st.caption(f"OCR engine used: {selected_engine}")
                        st.text_area("Extracted OCR text", value=extracted_text, height=120)
                        if looks_like_noisy_ocr(extracted_text):
                            st.warning(
                                "OCR output looks noisy. For Mandarin images, make sure "
                                "Expected image language is set to Mandarin and try a clearer "
                                "cropped image with only the text region."
                            )
                        if details:
                            st.dataframe(
                                pd.DataFrame(details),
                                hide_index=True,
                                use_container_width=True,
                            )
                    except Exception as exc:
                        show_model_error(st, exc)

        if extracted_text:
            with text_result:
                with st.spinner("Running DistilBERT text model..."):
                    try:
                        text_probs = predict_text(extracted_text, local_files_only=local_files_only)
                        render_prediction("DistilBERT text prediction", text_probs)
                    except Exception as exc:
                        show_model_error(st, exc)
        elif image_file is None:
            text_result.info("Upload an image or enter text to run DistilBERT.")

        if audio_probs is not None and text_probs is not None:
            st.divider()
            if fusion_method == "Equal Weighting (0.5/0.5)":
                fused_probs = fuse_equal(audio_probs, text_probs)
            elif fusion_method == "Accuracy-Weighted Fusion":
                fused_probs = fuse_accuracy_weighted(audio_probs, text_probs)
            else:
                fused_probs = fuse_confidence_adaptive(audio_probs, text_probs)

            render_prediction(f"Fusion prediction: {fusion_method}", fused_probs)

with tabs[1]:
    render_dashboard()
    st.subheader("OCR Machine Comparison")
    st.dataframe(OCR_COMPARISON, hide_index=True, use_container_width=True)

with tabs[2]:
    render_project_overview()

with tabs[3]:
    st.subheader("Expected Local Files")
    file_rows = [
        ("Wav2Vec2 checkpoint", AUDIO_MODEL_PATH),
        ("DistilBERT checkpoint", TEXT_MODEL_PATH),
        ("Hugging Face cache", HF_CACHE_DIR),
        ("Cached facebook/wav2vec2-base", HF_CACHE_DIR / "models--facebook--wav2vec2-base"),
        ("Cached distilbert-base-multilingual-cased", HF_CACHE_DIR / "models--distilbert-base-multilingual-cased"),
        ("Cached microsoft/trocr-base-printed", HF_CACHE_DIR / "models--microsoft--trocr-base-printed"),
        ("Fusion result JSON", RESULTS_PATH),
    ]
    st.dataframe(
        pd.DataFrame(
            {
                "Item": [row[0] for row in file_rows],
                "Path": [str(row[1]) for row in file_rows],
                "Found": [row[1].exists() for row in file_rows],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "For the first demo version, uploads are supported. Browser microphone "
        "recording and webcam capture can be added after the checkpoint pipeline "
        "runs cleanly in Streamlit."
    )

    st.subheader("Setup Commands")
    st.code(
        "pip install -r streamlit_requirements.txt\n"
        "python prepare_ui_models.py\n"
        "python -m streamlit run streamlit_app.py",
        language="powershell",
    )
