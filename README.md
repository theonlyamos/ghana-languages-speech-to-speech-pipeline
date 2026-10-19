# Ghana Speech-to-Speech Pipeline

A comprehensive pipeline for building multilingual Speech-to-Speech (S2S) AI systems for Ghanaian languages: **Akan (Twi/Fante)**, **Ewe**, **Ga**, and **Dagbani**.

## Architecture

```
+------------------+     +------------------+     +------------------+     +------------------+
|   THE EAR'S      |     |    THE EAR       |     |    THE BRAIN     |     |    THE MOUTH     |
|   TUNER (LID)    | --> |    (ASR)         | --> |    (Translation) | --> |    (TTS)         |
|   MMS-LID        |     |    Meta MMS      |     |    NLLB-200      |     |    XTTS v2       |
+------------------+     +------------------+     +------------------+     +------------------+
```

## Features

- **Automatic Language Detection**: Auto-detect which Ghanaian language is being spoken
- **Multi-language ASR**: Transcribe Akan, Ewe, Ga, Dagbani, and English
- **Cross-language Translation**: Translate between any supported language pair
- **High-quality TTS**: Natural speech synthesis with voice cloning support
- **Full S2S Pipeline**: End-to-end speech translation
- **Web Interface**: Gradio-based demo UI
- **REST API**: FastAPI endpoints for production deployment

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/theonlyamos/ghana-languages-speech-to-speech-pipeline
cd ghana-languages-speech-to-speech-pipeline
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install espeak-ng (for TTS)

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# Windows: Download from https://github.com/espeak-ng/espeak-ng/releases
```

## Quick Start

### Using the Jupyter Notebook

```bash
jupyter notebook ghana_s2s_pipeline.ipynb
```

The notebook is organized into 8 parts (Part 0–7):

0. Setup & System Verification
1. Dataset Download & Organization
2. Data Processing & Preparation
3. Model Training (ASR, TTS, Translation)
4. Unified Pipeline & Inference (including Language Detection)
5. Evaluation & Benchmarking
6. Deployment & Serving
7. Upload Models to HuggingFace

### Using the Pipeline Directly

```python
from utils.pipeline import GhanaS2SPipeline

# Initialize pipeline (includes Language ID model)
pipeline = GhanaS2SPipeline(
    device="cuda",
    load_lid=True,         # Enable automatic language detection
    load_asr=True,
    load_tts=True,
    load_translation=True
)

# Auto-detect language and transcribe
result = pipeline.listen("audio.wav", language="auto")
print(f"Detected: {result.language}, Text: {result.text}")

# Or specify language manually
result = pipeline.listen("audio.wav", language="aka")
print(f"Transcription: {result.text}")

# Detect language only (without transcription)
detected = pipeline.detect_language("audio.wav")
print(f"Language: {detected}")  # "aka", "ewe", "gaa", "dag", etc.

# Translate text
result = pipeline.think("Hello, how are you?", source_lang="eng", target_lang="aka")
print(f"Translation: {result.translated_text}")

# Synthesize speech
result = pipeline.speak("Maakye!", speaker="Twi_Speaker")
print(f"Audio saved to: {result.audio_path}")

# Full S2S pipeline with auto-detection
result = pipeline.run_pipeline(
    audio_input="unknown_language.wav",
    source_lang="auto",    # Auto-detect input language!
    target_lang="eng",
    translate=True
)
```

### Launch Web Interface

```python
from utils.serving import launch_gradio

launch_gradio(share=True)  # Opens browser with Gradio interface
```

### Run REST API

```bash
python -m uvicorn utils.serving:create_fastapi_app --host 0.0.0.0 --port 8000
```

### Run a Trained Model from the CLI

Run one-shot inference without writing any code:

```bash
# Full speech-to-speech: auto-detect language, transcribe, translate, synthesize
python scripts/infer.py audio.wav --target eng --output out.wav

# Transcribe only / detect language only / translate text only
python scripts/infer.py audio.wav --task transcribe
python scripts/infer.py audio.wav --task detect
python scripts/infer.py --task translate --text "Maakye" --source aka --target eng
```

`infer.py` auto-loads fine-tuned weights from `models/asr` and `models/tts` when
present (where training writes them) and falls back to the base MMS/NLLB/XTTS
models otherwise. Use `--asr-model-path` / `--tts-model-path` to point elsewhere,
`--base` to force the base models, `--no-speak` to skip synthesis, or `--json`
for machine-readable output.

The serving CLI accepts the same checkpoint flags, so the web UI / REST API serve
your fine-tuned models too:

```bash
python -m utils.serving --mode gradio    # or: --mode fastapi
# add --asr-model-path / --tts-model-path / --base to choose which weights to serve
```

## Natural Twi Voice (F5-TTS)

[F5-TTS](https://github.com/SWivid/F5-TTS) gives a far more natural voice than XTTS
and is simpler to fine-tune. Its pretrained alphabet already includes ɛ, ɔ and ŋ;
any other symbol in your data is added automatically.

```bash
# 1. BibleTTS Asante Twi (download in notebook Part 1.3) -> F5 dataset + reference clip
python scripts/train_tts_f5.py prepare --input data/raw/bibletis/asante_twi --name twi
#    add --limit 500 for a quick smoke run first

# 2. Fine-tune F5TTS_v1_Base (24 GB GPU; lower --batch-frames if you run out of memory)
python scripts/train_tts_f5.py train --name twi --epochs 20

# 3. Speak (the pipeline, web UI and API pick up models/tts_f5/<name> automatically)
python scripts/infer.py --task speak --text "Maakye, wo ho te sɛn?" --output twi.wav
```

Each voice lives in `models/tts_f5/<name>/` with its checkpoints, `vocab.txt` and a
`ref.wav`/`ref.txt` reference clip that sets the speaking style. Pass `--speaker-ref
other.wav` to clone a different voice. Training resumes from `model_last.pt` if
interrupted. Note: the pretrained F5 weights are CC-BY-NC (non-commercial).

## Training on Google Colab (CLI)

Don't have a 24GB GPU locally? Train on a remote Colab GPU straight from your
terminal with the official **[Google Colab CLI](https://github.com/googlecolab/google-colab-cli)**.
It provisions a GPU runtime, runs the training notebook on it, and pulls the
trained models back to your machine.

```bash
# Install + authenticate (one time)
uv tool install google-colab-cli
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

# Train (provision GPU -> install deps -> run notebook -> download models -> stop)
bash scripts/colab_train.sh --gpu L4
```

> **Windows users:** the Colab CLI runs on Linux/macOS only — use **WSL2**.
> **GPU access is tier-gated** (T4/L4/A100 generally need Colab Pro/Pro+/Pay-As-You-Go),
> and an unstopped session burns compute units until a 24h cap, so always `colab stop`.

**Full guide:** [COLAB_CLI.md](COLAB_CLI.md) — authentication, GPU selection,
handling the large datasets, one-shot `colab run` jobs, and a command cheat-sheet.

## Project Structure

```
ghana_sts_model/
├── ghana_s2s_pipeline.ipynb  # Main comprehensive notebook
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
├── SETUP.md                   # Local environment setup guide
├── COLAB_CLI.md               # Train on Google Colab GPUs via the CLI
├── scripts/
│   ├── colab_train.sh        # One-command Colab CLI training workflow
│   ├── train_tts_f5.py       # Prepare + fine-tune a natural F5-TTS voice
│   └── infer.py              # One-shot CLI inference (trained or base models)
├── utils/
│   ├── __init__.py
│   ├── data_processing.py    # Dataset utilities
│   ├── pipeline.py           # GhanaS2SPipeline class
│   └── serving.py            # Gradio/FastAPI helpers
├── data/                      # Downloaded datasets (created on first run)
│   ├── raw/
│   └── processed/
├── models/                    # Trained models (created on first run)
│   ├── asr/
│   └── tts/
└── outputs/                   # Generated audio files
```

## Datasets

The pipeline primarily uses the following datasets:

| Dataset                                                                                   | Languages                           | Size   | Use          |
| ----------------------------------------------------------------------------------------- | ----------------------------------- | ------ | ------------ |
| [UGSpeechData](https://www.scidb.cn/en/detail?dataSetId=bbd6baee3acf43bbbc4fe25e21077c8a) | Akan, Ewe, Dagbani, Dagaare, Ikposo | ~336GB | ASR Training |
| [FISD](https://github.com/Ashesi-Org/Financial-Inclusion-Speech-Dataset)                  | Ga, Fante, Akuapem Twi, Asante Twi  | ~1.2GB | Domain ASR   |
| [BibleTTS](http://www.openslr.org/129/)                                                   | Asante Twi, Akuapem Twi, Ewe, Hausa | ~50GB  | TTS Training |

- **UGSpeechData**: Multilingual speech dataset of Ghanaian languages ([SciDB](https://www.scidb.cn/en/detail?dataSetId=bbd6baee3acf43bbbc4fe25e21077c8a)).
- **FISD**: Financial Inclusion Speech Dataset by Ashesi University and Nokwary Technologies (~148 hours, CC-BY-4.0) ([GitHub](https://github.com/Ashesi-Org/Financial-Inclusion-Speech-Dataset)).

## Language Codes

| Language   | MMS (ASR) | NLLB (Translation) | TTS Speaker     | Auto-Detect |
| ---------- | --------- | ------------------ | --------------- | ----------- |
| Akan (Twi) | aka       | aka_Latn           | Twi_Speaker     | Yes         |
| Ewe        | ewe       | ewe_Latn           | Ewe_Speaker     | Yes         |
| Ga         | gaa       | gaa_Latn           | Ga_Speaker      | Yes         |
| Dagbani    | dag       | dag_Latn           | Dagbani_Speaker | Yes         |
| English    | eng       | eng_Latn           | -               | Yes         |

Use `language="auto"` to automatically detect the input language.

## Hardware Requirements

- **GPU**: NVIDIA RTX 3090/4090 (24GB VRAM) recommended
- **RAM**: 32GB+ recommended
- **Storage**: 250GB+ free for full datasets

## Cheap GPU VPS / Cloud Options

For running the pipeline without a local GPU, these platforms offer affordable GPU instances with 24GB+ VRAM. Prices change; check each provider’s pricing page for current rates.

> **Prefer Google Colab?** You can also train directly on Colab GPUs from your
> terminal with the Google Colab CLI — see [Training on Google Colab (CLI)](#training-on-google-colab-cli)
> and [COLAB_CLI.md](COLAB_CLI.md).

| Provider | Best for | 24GB GPU (e.g. RTX 3090) | Notes |
| -------- | -------- | ------------------------- | ----- |
| **[Vast.ai](https://vast.ai/)** | Lowest cost | ~$0.08–0.18/hr | Marketplace pricing; often 3–6× cheaper than big cloud. Per-minute billing, many RTX 3090/4090 options. [Pricing](https://vast.ai/pricing) |
| **[RunPod](https://www.runpod.io/)** | Reliability + serverless | ~$0.14–0.44/hr (pods); serverless per-second | Fixed pricing, Docker-friendly, serverless for inference. [Pricing](https://www.runpod.io/pricing-page) |
| **[DigitalOcean](https://www.digitalocean.com/pricing/gpu-droplets)** | Simplicity | RTX 4000 20GB ~$0.76/hr; L40S 48GB ~$1.57/hr | Predictable pricing, good for small teams. 20GB may work with reduced batch size. |
| **[Lambda Labs](https://lambdalabs.com/)** | Stability | ~$0.50+/hr | Production-style infra, 1-click Jupyter; often higher cost and limited availability. |
| **[Vultr](https://www.vultr.com/products/cloud-gpu/)** | Global regions | Varies | NVIDIA partner, many regions; check current GPU offerings. |

**Suggestions for this project**

- **Training / long runs**: [Vast.ai](https://vast.ai/) or [RunPod](https://www.runpod.io/) spot/pods — filter for RTX 3090 (24GB) or A100 (40/80GB) and attach enough storage (250GB+ for full datasets).
- **Inference / API**: RunPod [Serverless](https://www.runpod.io/product/serverless) (pay per second, no idle cost) or a small always-on pod on Vast.ai/RunPod.
- **Budget-first**: Vast.ai usually has the lowest hourly rates; use Interruptible/spot for non-urgent jobs.

Storage (250GB+ for full data) is billed separately on most platforms (e.g. RunPod network volumes ~$0.07/GB/mo; Vast.ai varies by host). Prefer regions close to your users for lower latency.

## API Endpoints

| Method | Endpoint                | Description                           |
| ------ | ----------------------- | ------------------------------------- |
| POST   | `/api/transcribe`       | Speech to text (supports `auto` lang) |
| POST   | `/api/translate`        | Text translation                      |
| POST   | `/api/synthesize`       | Text to speech                        |
| POST   | `/api/speech-to-speech` | Full S2S pipeline (supports `auto`)   |
| GET    | `/api/languages`        | List supported languages              |
| GET    | `/health`               | Health check                          |

**Example with auto-detection:**

```bash
curl -X POST "http://localhost:8000/api/speech-to-speech" \
  -F "audio=@input.wav" \
  -F "source_lang=auto" \
  -F "target_lang=eng"
```

## Configuration

Edit `config.py` to customize:

```python
# Sample mode for quick testing
config.dataset.sample_mode = True
config.dataset.sample_size = 1000

# Target languages
config.dataset.languages = ["aka", "ewe", "gaa", "dag"]

# Training parameters
config.asr.batch_size = 4
config.asr.learning_rate = 1e-4
config.tts.epochs = 10
```

## Citations

```bibtex
@article{pratap2023mms,
  title={Scaling Speech Technology to 1,000+ Languages},
  author={Pratap, Vineel and others},
  journal={arXiv preprint arXiv:2305.13516},
  year={2023}
}

@article{costa2022nllb,
  title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  author={Costa-jussà, Marta R and others},
  journal={arXiv preprint arXiv:2207.04672},
  year={2022}
}
```

## License

This project uses models and datasets with various licenses:

- MMS: CC-BY-NC 4.0
- NLLB: CC-BY-NC 4.0
- BibleTTS: CC-BY-SA 4.0
- FISD: CC-BY-4.0
- UGSpeechData: Academic use

Please check individual dataset/model licenses for commercial use.

## Contributing

Contributions are welcome! Please open an issue or pull request for:

- Bug fixes
- New language support
- Performance improvements
- Documentation updates
