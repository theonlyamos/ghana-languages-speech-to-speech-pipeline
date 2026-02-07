# Ghana Speech-to-Speech Pipeline

A comprehensive pipeline for building multilingual Speech-to-Speech (S2S) AI systems for Ghanaian languages: **Akan (Twi/Fante)**, **Ewe**, **Ga**, and **Dagbani**.

## Architecture

```
+------------------+     +------------------+     +------------------+
|    THE EAR       |     |    THE BRAIN     |     |    THE MOUTH     |
|    (ASR)         | --> |    (Translation) | --> |    (TTS)         |
|    Meta MMS      |     |    NLLB-200      |     |    XTTS v2       |
+------------------+     +------------------+     +------------------+
```

## Features

- **Multi-language ASR**: Transcribe Akan, Ewe, Ga, Dagbani, and English
- **Cross-language Translation**: Translate between any supported language pair
- **High-quality TTS**: Natural speech synthesis with voice cloning support
- **Full S2S Pipeline**: End-to-end speech translation
- **Web Interface**: Gradio-based demo UI
- **REST API**: FastAPI endpoints for production deployment

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ghana_sts_model
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

The notebook contains 6 main parts:
1. Setup & System Verification
2. Dataset Download & Organization
3. Data Processing & Preparation
4. Model Training (ASR, TTS, Translation)
5. Unified Pipeline & Inference
6. Deployment & Serving

### Using the Pipeline Directly

```python
from utils.pipeline import GhanaS2SPipeline

# Initialize pipeline
pipeline = GhanaS2SPipeline(
    device="cuda",
    load_asr=True,
    load_tts=True,
    load_translation=True
)

# Transcribe audio
result = pipeline.listen("audio.wav", language="aka")
print(f"Transcription: {result.text}")

# Translate text
result = pipeline.think("Hello, how are you?", source_lang="eng", target_lang="aka")
print(f"Translation: {result.translated_text}")

# Synthesize speech
result = pipeline.speak("Maakye!", speaker="Twi_Speaker")
print(f"Audio saved to: {result.audio_path}")

# Full S2S pipeline
result = pipeline.run_pipeline(
    audio_input="english_audio.wav",
    source_lang="eng",
    target_lang="aka",
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

## Project Structure

```
ghana_sts_model/
├── ghana_s2s_pipeline.ipynb  # Main comprehensive notebook
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
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

The pipeline uses data from:

| Dataset | Languages | Size | Use |
|---------|-----------|------|-----|
| [UGSpeechData](https://www.scidb.cn/en/detail?dataSetId=bbd6baee3acf43bbbc4fe25e21077c8a) | Akan, Ewe, Dagbani, Dagaare, Ikposo | ~336GB | ASR Training |
| [BibleTTS](http://www.openslr.org/129/) | Asante Twi, Akuapem Twi, Ewe | ~50GB | TTS Training |
| [FISD](https://adr.ashesi.edu.gh/datasets) | Ga, Fante, Twi | ~15GB | Domain ASR |

## Language Codes

| Language | MMS (ASR) | NLLB (Translation) | TTS Speaker |
|----------|-----------|-------------------|-------------|
| Akan (Twi) | aka | aka_Latn | Twi_Speaker |
| Ewe | ewe | ewe_Latn | Ewe_Speaker |
| Ga | gaa | gaa_Latn | Ga_Speaker |
| Dagbani | dag | dag_Latn | Dagbani_Speaker |
| English | eng | eng_Latn | - |

## Hardware Requirements

- **GPU**: NVIDIA RTX 3090/4090 (24GB VRAM) recommended
- **RAM**: 32GB+ recommended
- **Storage**: 250GB+ free for full datasets

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/transcribe` | Speech to text |
| POST | `/api/translate` | Text translation |
| POST | `/api/synthesize` | Text to speech |
| POST | `/api/speech-to-speech` | Full S2S pipeline |
| GET | `/api/languages` | List supported languages |
| GET | `/health` | Health check |

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
  title={No Language Left Behind},
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
- UGSpeechData: Academic use

Please check individual dataset/model licenses for commercial use.

## Contributing

Contributions are welcome! Please open an issue or pull request for:
- Bug fixes
- New language support
- Performance improvements
- Documentation updates
