"""
Ghana Speech-to-Speech Pipeline Configuration
=============================================
Central configuration file for dataset paths, model settings, and training parameters.
Optimized for RTX 3090/4090 (24GB VRAM).
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# BASE PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
@dataclass
class DatasetConfig:
    """Configuration for dataset downloads and processing."""
    
    # Toggle for development mode (uses smaller subsets)
    sample_mode: bool = True
    sample_size: int = 1000  # Number of samples per language in sample mode
    
    # Target languages (ISO 639-3 codes)
    languages: List[str] = field(default_factory=lambda: ["aka", "ewe", "gaa", "dag"])
    
    # Language display names
    language_names: Dict[str, str] = field(default_factory=lambda: {
        "aka": "Akan (Twi/Fante)",
        "ewe": "Ewe",
        "gaa": "Ga",
        "dag": "Dagbani",
        "dga": "Dagaare",
        "kpo": "Ikposo"
    })
    
    # UGSpeechData URLs (Science Data Bank)
    ugspeechdata_urls: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "akan": {
            "audio": "https://download.scidb.cn/download?fileId=c339943da85188e1fe636acf3673e55f&path=/V6/Data/Akan.tar&fileName=Akan.tar",
            "metadata": "https://download.scidb.cn/download?fileId=43351b4a85e16eb57115668c0448f990&path=/V6/Data/Metadata/Akan.xlsx&fileName=Akan.xlsx",
            "transcribed": "https://download.scidb.cn/download?fileId=8083b8dc094efd29273d8ace1ebae577&path=/V6/Transcribed%20data/Akan.zip&fileName=Akan.zip"
        },
        "ewe": {
            "audio": "https://download.scidb.cn/download?fileId=0645a0503fa260de92a676b42db13af6&path=/V6/Data/Ewe.tar&fileName=Ewe.tar",
            "metadata": "https://download.scidb.cn/download?fileId=c07c35602c9ed4d3dffbc4622d64561b&path=/V6/Data/Metadata/Ewe.xlsx&fileName=Ewe.xlsx",
            "transcribed": "https://download.scidb.cn/download?fileId=ad932712f012b8e55d07e9bd6cb7b6b0&path=/V6/Transcribed%20data/Ewe.zip&fileName=Ewe.zip"
        },
        "dagbani": {
            "audio": "https://download.scidb.cn/download?fileId=7bd3760f0233d938803a421567693ffb&path=/V6/Data/Dagbani.tar&fileName=Dagbani.tar",
            "metadata": "https://download.scidb.cn/download?fileId=b70a70f43695304928879f12520d7b61&path=/V6/Data/Metadata/Dagbani.xlsx&fileName=Dagbani.xlsx",
            "transcribed": "https://download.scidb.cn/download?fileId=21cc98210abc81a45e58e707448037ac&path=/V6/Transcribed%20data/Dagbani.zip&fileName=Dagbani.zip"
        },
        "dagaare": {
            "audio": "https://download.scidb.cn/download?fileId=7861c9a2a22397a941158b22a04f3a4c&path=/V6/Data/Dagaare.tar&fileName=Dagaare.tar",
            "metadata": "https://download.scidb.cn/download?fileId=d2e30f2c4955b799b5a1f92d1df5f714&path=/V6/Data/Metadata/Dagaare.xlsx&fileName=Dagaare.xlsx",
            "transcribed": "https://download.scidb.cn/download?fileId=6e829321b1d63b1080d737f83b81d18d&path=/V6/Transcribed%20data/Dagaare.zip&fileName=Dagaare.zip"
        },
        "ikposo": {
            "audio": "https://download.scidb.cn/download?fileId=4299bd8df6591e40bf9bea5a7b2bf0ea&path=/V6/Data/Ikposo.tar&fileName=Ikposo.tar",
            "metadata": "https://download.scidb.cn/download?fileId=997a275d77e962520ee3b59773ee0fbb&path=/V6/Data/Metadata/Ikposo.xlsx&fileName=Ikposo.xlsx",
            "transcribed": "https://download.scidb.cn/download?fileId=410746d6d90c2d6730dc4f7811dccec3&path=/V6/Transcribed%20data/Ikposo.zip&fileName=Ikposo.zip"
        }
    })
    
    # BibleTTS URLs (OpenSLR)
    bibletis_base_url: str = "https://www.openslr.org/resources/129"
    bibletis_files: Dict[str, str] = field(default_factory=lambda: {
        "asante_twi": "asante-twi.tgz",
        "akuapem_twi": "akuapem-twi.tgz",
        "ewe": "ewe.tgz",
        "hausa": "hausa.tgz"
    })
    
    # FISD (Ashesi Financial Inclusion Speech Dataset) URLs
    fisd_urls: Dict[str, List[str]] = field(default_factory=lambda: {
        "ga": [
            "https://fisd-dataset.s3.amazonaws.com/fisd-ga-90p.zip",
            "https://fisd-dataset.s3.amazonaws.com/fisd-ga-10p.zip"
        ],
        "fante": [
            "https://fisd-dataset.s3.amazonaws.com/fisd-fanti-90p.zip",
            "https://fisd-dataset.s3.amazonaws.com/fisd-fanti-10p.zip"
        ],
        "akuapim_twi": [
            "https://fisd-dataset.s3.amazonaws.com/fisd-akuapim-twi-90p.zip",
            "https://fisd-dataset.s3.amazonaws.com/fisd-akuapim-twi-10p.zip"
        ],
        "asante_twi": [
            "https://fisd-dataset.s3.amazonaws.com/fisd-asanti-twi-90p.zip",
            "https://fisd-dataset.s3.amazonaws.com/fisd-asanti-twi-10p.zip"
        ]
    })
    
    # Data directories
    raw_data_dir: Path = field(default_factory=lambda: DATA_DIR / "raw")
    processed_data_dir: Path = field(default_factory=lambda: DATA_DIR / "processed")
    asr_data_dir: Path = field(default_factory=lambda: DATA_DIR / "processed" / "asr")
    tts_data_dir: Path = field(default_factory=lambda: DATA_DIR / "processed" / "tts")


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
@dataclass
class ASRConfig:
    """Configuration for ASR (Automatic Speech Recognition) model."""
    
    # Base model
    model_id: str = "facebook/mms-1b-all"
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Quantization
    use_8bit: bool = True
    
    # Audio settings
    sample_rate: int = 16000
    
    # Training settings (optimized for RTX 3090/4090)
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 5
    warmup_steps: int = 100
    max_steps: int = 2000
    fp16: bool = True
    
    # Checkpointing
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 50
    save_total_limit: int = 3
    
    # Output
    output_dir: Path = field(default_factory=lambda: MODEL_DIR / "asr")


@dataclass
class TTSConfig:
    """Configuration for TTS (Text-to-Speech) model."""
    
    # Base model
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    # Audio settings
    sample_rate: int = 22050
    
    # Speaker configuration
    speakers: Dict[str, str] = field(default_factory=lambda: {
        "aka": "Twi_Speaker",
        "ewe": "Ewe_Speaker",
        "gaa": "Ga_Speaker",
        "dag": "Dagbani_Speaker"
    })
    
    # Training settings (optimized for RTX 3090/4090)
    batch_size: int = 2
    num_loader_workers: int = 4
    epochs: int = 10
    learning_rate: float = 5e-6
    
    # We use 'en' as carrier language for XTTS (trick for unsupported languages)
    carrier_language: str = "en"
    
    # Freeze settings
    train_gpt: bool = True
    train_hifi_gan: bool = False
    train_speaker_encoder: bool = False
    
    # Output
    output_dir: Path = field(default_factory=lambda: MODEL_DIR / "tts")


@dataclass
class TranslationConfig:
    """Configuration for Translation (Machine Translation) model."""
    
    # Model
    model_id: str = "facebook/nllb-200-distilled-600M"  # Use "3.3B" for better quality
    
    # NLLB language codes
    lang_codes: Dict[str, str] = field(default_factory=lambda: {
        "aka": "aka_Latn",  # Akan/Twi
        "twi": "aka_Latn",  # Alias
        "ewe": "ewe_Latn",
        "gaa": "gaa_Latn",  # Ga
        "dag": "dag_Latn",  # Dagbani
        "eng": "eng_Latn",
        "en": "eng_Latn"   # Alias
    })
    
    # MMS language codes (ISO 639-3)
    mms_lang_codes: Dict[str, str] = field(default_factory=lambda: {
        "akan": "aka",
        "twi": "aka",
        "ewe": "ewe",
        "ga": "gaa",
        "dagbani": "dag",
        "english": "eng"
    })
    
    # Generation settings
    max_length: int = 200
    num_beams: int = 4
    
    # Quantization for memory efficiency
    use_8bit: bool = False  # NLLB-600M fits in memory without quantization


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================
@dataclass
class PipelineConfig:
    """Configuration for the unified S2S pipeline."""
    
    # Device settings
    device: str = "cuda"  # or "cpu"
    
    # Component configs
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    
    # Pipeline settings
    default_source_lang: str = "aka"
    default_target_lang: str = "eng"
    
    # Voice cloning
    reference_audio_duration: float = 6.0  # seconds
    
    # Output
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)


# ============================================================================
# SERVING CONFIGURATION
# ============================================================================
@dataclass
class ServingConfig:
    """Configuration for deployment (Gradio/FastAPI)."""
    
    # Gradio settings
    gradio_share: bool = False
    gradio_port: int = 7860
    
    # FastAPI settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Audio settings
    max_audio_duration: float = 30.0  # seconds
    allowed_audio_formats: List[str] = field(default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg"])


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # ASR metrics
    compute_wer: bool = True
    compute_cer: bool = True
    
    # TTS metrics
    compute_mos: bool = False  # Requires MOSNet
    
    # Translation metrics
    compute_bleu: bool = True
    
    # Benchmark datasets
    use_fleurs: bool = True
    use_common_voice: bool = True


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================
@dataclass
class Config:
    """Master configuration class."""
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def __post_init__(self):
        """Create necessary directories."""
        dirs = [
            self.dataset.raw_data_dir,
            self.dataset.processed_data_dir,
            self.dataset.asr_data_dir,
            self.dataset.tts_data_dir,
            self.asr.output_dir,
            self.tts.output_dir,
            self.pipeline.output_dir
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# Create global config instance
config = Config()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_language_code(lang: str, format: str = "mms") -> str:
    """
    Convert language name/code to the required format.
    
    Args:
        lang: Language name or code (e.g., "twi", "akan", "aka")
        format: Target format - "mms" for ASR, "nllb" for translation
    
    Returns:
        Converted language code
    """
    lang = lang.lower().strip()
    
    if format == "mms":
        # Map to ISO 639-3 codes
        mapping = {
            "akan": "aka", "twi": "aka", "fante": "aka",
            "ewe": "ewe",
            "ga": "gaa",
            "dagbani": "dag",
            "dagaare": "dga",
            "english": "eng", "en": "eng"
        }
        return mapping.get(lang, lang)
    
    elif format == "nllb":
        # Map to NLLB codes
        mapping = {
            "akan": "aka_Latn", "twi": "aka_Latn", "aka": "aka_Latn",
            "ewe": "ewe_Latn",
            "ga": "gaa_Latn", "gaa": "gaa_Latn",
            "dagbani": "dag_Latn", "dag": "dag_Latn",
            "english": "eng_Latn", "eng": "eng_Latn", "en": "eng_Latn"
        }
        return mapping.get(lang, f"{lang}_Latn")
    
    return lang


def print_config():
    """Print current configuration summary."""
    print("=" * 60)
    print("Ghana Speech-to-Speech Pipeline Configuration")
    print("=" * 60)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nSample Mode: {config.dataset.sample_mode}")
    print(f"Target Languages: {config.dataset.languages}")
    print(f"\nASR Model: {config.asr.model_id}")
    print(f"TTS Model: {config.tts.model_name}")
    print(f"Translation Model: {config.translation.model_id}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
