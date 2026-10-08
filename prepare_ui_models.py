from pathlib import Path


ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = ROOT / "hf_cache"


def main():
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        Wav2Vec2Processor,
    )

    HF_CACHE_DIR.mkdir(exist_ok=True)

    print("Caching facebook/wav2vec2-base processor...")
    Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base",
        cache_dir=str(HF_CACHE_DIR),
    )

    print("Caching distilbert-base-multilingual-cased tokenizer and base model...")
    DistilBertTokenizer.from_pretrained(
        "distilbert-base-multilingual-cased",
        cache_dir=str(HF_CACHE_DIR),
    )
    DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased",
        num_labels=3,
        cache_dir=str(HF_CACHE_DIR),
        ignore_mismatched_sizes=True,
    )

    print("Caching microsoft/trocr-base-printed...")
    TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-printed",
        cache_dir=str(HF_CACHE_DIR),
    )
    VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed",
        cache_dir=str(HF_CACHE_DIR),
    )

    print("Done. You can now enable local-cache-only mode for offline demos.")


if __name__ == "__main__":
    main()
