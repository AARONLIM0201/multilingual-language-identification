# Multilingual Low-Resource Language Identification Using Audio and OCR-Based Text

A multimodal language identification system for **English, Malay, and Mandarin**, combining spoken audio and OCR-based text to improve language classification under noisy and low-resource conditions.

> **Final Year Project — Multimedia University (MMU)**
> Bachelor of Computer Science (Hons), Data Science / Data Analytics
> **Student:** Aaron Lim Cjun Shien
> **Supervisor:** Dr. S. Prabha Kumaresan

---

## 📌 Project Overview

Malaysia is a multilingual environment where English, Malay, and Mandarin are frequently used. Identifying languages from real-world inputs can become challenging when the available data is limited, audio quality is poor, or text is provided as an image rather than directly as digital text.

This project develops a **multimodal language identification system** that accepts:

* 🎙️ **Audio input** — spoken language
* 🖼️ **Image input** — text extracted using Optical Character Recognition (OCR)

The system first evaluates audio and text independently before combining their predictions through **late fusion**.

The final system identifies one of three languages:

* 🇬🇧 English
* 🇲🇾 Malay
* 🇨🇳 Mandarin

The project compares lightweight baseline models with transformer-based models and evaluates several multimodal fusion strategies.

---

## 🎯 Objectives

1. Develop an audio-based language identification model for English, Malay, and Mandarin.
2. Develop an OCR-based text language identification pipeline using noisy text images.
3. Compare traditional/lightweight models with transformer-based approaches.
4. Investigate multimodal late-fusion strategies for combining audio and text predictions.
5. Analyse model performance across different languages and input conditions.
6. Develop an interactive demonstration system for multimodal language identification.

---

## 🏗️ System Architecture

The overall workflow consists of two independent modalities followed by multimodal fusion.

```text
                         Input
                           │
              ┌────────────┴────────────┐
              │                         │
          🎙️ Audio                   🖼️ Image
              │                         │
        Preprocessing                  OCR
              │                         │
              ▼                         ▼
        CNN / wav2vec 2.0        TrOCR / EasyOCR
              │                         │
              ▼                         ▼
       Audio Prediction          Extracted Text
                                        │
                                        ▼
                                FastText / DistilBERT
                                        │
                                        ▼
                                 Text Prediction
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                    Late Fusion
                           │
             ┌─────────────┼─────────────┐
             │             │             │
           Equal      Confidence      Accuracy
          Weighting     Adaptive       Weighted
             │             │             │
             └─────────────┴─────────────┘
                           │
                           ▼
                  Final Language
                    Prediction
                           │
                  English / Malay /
                     Mandarin
```

---

## 📊 Dataset

### Audio Dataset

Audio samples were obtained from **VoxLingua107** and prepared for multilingual language identification.

The main audio experiments used approximately **1 hour per language** for the standalone experiments, with an additional **7.5-hour audio dataset** used for the multimodal system.

| Parameter                 | Configuration            |
| ------------------------- | ------------------------ |
| Languages                 | English, Malay, Mandarin |
| Sampling rate             | 16 kHz                   |
| CNN segment length        | 3 seconds                |
| wav2vec 2.0 input         | 10 seconds               |
| CNN features              | 128 Mel bins             |
| FFT size                  | 1024                     |
| Hop length                | 512                      |
| Train / Validation / Test | 70% / 15% / 15%          |

### Text Image Dataset

Text samples were collected from Wikipedia and rendered into images to simulate image-based text input.

The dataset was designed to evaluate OCR performance under both clean and noisy conditions.

Noise and image degradation included:

* Gaussian noise
* Salt-and-pepper noise
* Motion blur
* JPEG compression
* Occlusion
* Brightness and contrast variations

---

## 🤖 Models

### Audio Modality

| Model       | Description                             |   Accuracy |
| ----------- | --------------------------------------- | ---------: |
| CNN         | Convolutional Neural Network baseline   | **73.56%** |
| wav2vec 2.0 | Pre-trained speech representation model | **93.91%** |

### Text Modality

| Model      | Description                              |   Accuracy |
| ---------- | ---------------------------------------- | ---------: |
| FastText   | Lightweight text classification baseline | **98.44%** |
| DistilBERT | Transformer-based text classifier        | **99.73%** |

### OCR Evaluation

Two OCR engines were evaluated before selecting the final OCR approach.

| OCR Engine  |    English |      Malay |   Mandarin |
| ----------- | ---------: | ---------: | ---------: |
| TrOCR       |     62.24% |     46.65% |      2.56% |
| **EasyOCR** | **87.90%** | **87.62%** | **76.82%** |

**EasyOCR was selected as the final OCR engine** because it provided substantially stronger performance across all three languages.

---

## 🔀 Multimodal Fusion Results

The final multimodal system combines audio and OCR-derived text predictions using late fusion.

| Method                     | Modality         |   Accuracy |
| -------------------------- | ---------------- | ---------: |
| Audio Only                 | Audio            |     94.16% |
| Text Only                  | OCR Text         |     97.28% |
| Confidence-Adaptive Fusion | Audio + Text     |     99.01% |
| Accuracy-Weighted Fusion   | Audio + Text     |     97.86% |
| **Equal Weighting Fusion** | **Audio + Text** | **99.18%** |

### Best Result

**Equal Weighting Fusion achieved 99.18% accuracy**, providing the strongest overall performance among the evaluated fusion strategies.

Per-language performance for the Equal Weighting strategy:

| Language | Accuracy |
| -------- | -------: |
| English  |   99.26% |
| Malay    |   98.52% |
| Mandarin |   99.75% |

---

## 🔎 Error Analysis

An error analysis was performed on the multimodal test predictions to investigate how the two modalities complement each other.

| Prediction Outcome      |  Samples |
| ----------------------- | -------: |
| Both modalities correct |     1113 |
| Only text correct       |       69 |
| Only audio correct      |       31 |
| Both modalities wrong   |        2 |
| **Total**               | **1215** |

The analysis shows that the two modalities provide complementary information. Text successfully corrected a number of audio errors, while audio corrected cases where OCR-based text extraction was incorrect.

The remaining errors were mainly associated with:

* Acoustic similarity between languages
* Image blur and compression
* Pixel-level noise affecting OCR
* Increased OCR difficulty for Mandarin characters under severe image degradation

---

## 🖥️ Interactive Demonstration

The project includes a **Streamlit-based demonstration interface** that allows users to interact with the trained language identification models.

The application supports the project's multimodal workflow:

```text
Audio Input ──► Audio Model ──┐
                              ├──► Fusion ──► Language
Image Input ──► OCR ──► Text Model ─┘          Prediction
```

The main application is:

```text
streamlit_app.py
```

---

## 🛠️ Technologies Used

### Programming & Development

* Python
* Jupyter Notebook
* Streamlit

### Machine Learning / Deep Learning

* PyTorch
* Hugging Face Transformers
* FastText
* scikit-learn

### Audio Processing

* Librosa
* wav2vec 2.0
* Mel-spectrogram features

### OCR & Image Processing

* EasyOCR
* TrOCR
* Pillow
* OpenCV
* NumPy

### Data Processing & Visualisation

* pandas
* NumPy
* Matplotlib
* Seaborn

---

## 📁 Repository Structure

```text
multilingual-language-identification/
│
├── CNN/
│   ├── CNN.ipynb
│   ├── CNN_7p5h.ipynb
│   ├── Cnn+FastText.ipynb
│   └── *.json
│
├── docs/
│   └── Final_Year_Project_Interim_Report.docx
│
├── *.ipynb
│   ├── FastText.ipynb
│   ├── DistilBERT.ipynb
│   ├── wav2vec2 (audio Transformer).ipynb
│   ├── 7.5h wav2vec2 (audio Transformer).ipynb
│   ├── OCR_Image_Noise.ipynb
│   ├── Wav2Vec2+DistilBERT.ipynb
│   └── Multimodal Language Identifier.ipynb
│
├── streamlit_app.py
├── prepare_ui_models.py
├── measure_env_info.py
├── measure_gpu.py
│
├── *.csv
├── *.json
├── *.png
│
├── streamlit_requirements.txt
├── README.md
└── .gitignore
```

Large datasets, trained model weights, virtual environments, cache files, and private project documents are intentionally excluded from the repository.

---

## 🚀 Running the Demonstration

### 1. Clone the repository

```bash
git clone https://github.com/AARONLIM0201/multilingual-language-identification.git
cd multilingual-language-identification
```

### 2. Create a virtual environment

```bash
python -m venv fypenv
```

Activate it on Windows:

```powershell
.\fypenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r streamlit_requirements.txt
```

### 4. Start the Streamlit application

```bash
streamlit run streamlit_app.py
```

The application will then provide a local web interface for the demonstration.

> **Note:** The repository does not include large trained model weights or datasets. These files are excluded to keep the repository lightweight and to avoid distributing large model artifacts.

---

## 📈 Key Findings

* **wav2vec 2.0** substantially outperformed the CNN audio baseline.
* **DistilBERT** achieved the highest text-only classification accuracy.
* **EasyOCR** was more effective than TrOCR for the evaluated multilingual image dataset.
* Audio and text provide **complementary information** for language identification.
* Multimodal late fusion improved performance over the individual modalities.
* **Equal Weighting Fusion achieved the best overall accuracy of 99.18%.**
* The multimodal system achieved particularly strong performance for Mandarin, reaching **99.75% accuracy** under the Equal Weighting strategy.

---

## 🏆 Achievement

**Gold Medal — INVENTX MMU 2026**

This project was presented as a Final Year Project at Multimedia University and received a **Gold Medal at INVENTX MMU 2026**.

---

## 📄 Project Documentation

The project documentation is available in:

```text
docs/Final_Year_Project_Interim_Report.docx
```

The report contains detailed information on the project background, methodology, dataset preparation, model development, experiments, results, analysis, discussion, limitations, and recommendations.

---

## 👨‍💻 Author

**Aaron Lim Cjun Shien**

Bachelor of Computer Science (Hons)
Multimedia University (MMU)

### Supervisor

**Dr. S. Prabha Kumaresan**

Faculty of Computing and Informatics
Multimedia University

---

## 📜 License

This project is intended primarily for **academic and portfolio purposes**.

Please refer to the repository contents and associated dataset/model licenses before redistributing any third-party datasets, pretrained models, or resources used by this project.

