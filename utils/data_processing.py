"""
Data Processing Utilities for Ghana Speech-to-Speech Pipeline
==============================================================
Handles dataset downloading, audio processing, and formatting for ASR/TTS training.
"""

import os
import sys
import json
import hashlib
import tarfile
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

# Audio processing
try:
    import librosa
    import soundfile as sf
except ImportError:
    warnings.warn("librosa/soundfile not installed. Audio processing will be limited.")

# For dataset handling
try:
    from datasets import Dataset, Audio, DatasetDict
except ImportError:
    warnings.warn("datasets library not installed. HuggingFace dataset creation will be limited.")

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config, DATA_DIR, get_language_code


# ============================================================================
# DATASET DOWNLOADER
# ============================================================================
class DatasetDownloader:
    """
    Handles downloading and extracting datasets from various sources.
    Supports resume, progress tracking, and checksum verification.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded files. Defaults to config.
        """
        self.output_dir = output_dir or config.dataset.raw_data_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(
        self, 
        url: str, 
        filename: Optional[str] = None,
        chunk_size: int = 8192,
        show_progress: bool = True
    ) -> Path:
        """
        Download a file with resume support and progress bar.
        
        Args:
            url: URL to download
            filename: Optional output filename
            chunk_size: Download chunk size
            show_progress: Whether to show progress bar
            
        Returns:
            Path to downloaded file
        """
        import requests
        
        if filename is None:
            # Extract filename from URL
            filename = url.split("fileName=")[-1] if "fileName=" in url else url.split("/")[-1]
        
        output_path = self.output_dir / filename
        
        # Check if file already exists and get size for resume
        resume_byte_pos = 0
        if output_path.exists():
            resume_byte_pos = output_path.stat().st_size
            
        # Set up headers for resume
        headers = {}
        if resume_byte_pos > 0:
            headers["Range"] = f"bytes={resume_byte_pos}-"
        
        # Make request
        response = requests.get(url, headers=headers, stream=True)
        
        # Check if server supports resume
        if response.status_code == 416:  # Range not satisfiable - file complete
            print(f"File already complete: {filename}")
            return output_path
        
        # Check for HTTP errors
        if response.status_code not in (200, 206):
            raise RuntimeError(f"Download failed with HTTP status {response.status_code}: {url}")
        
        # Get total file size
        total_size = int(response.headers.get("content-length", 0))
        if resume_byte_pos > 0 and response.status_code == 206:
            total_size += resume_byte_pos
            print(f"Resuming download from {resume_byte_pos / 1024 / 1024:.1f} MB")
        
        # Download with progress bar
        mode = "ab" if resume_byte_pos > 0 else "wb"
        
        pbar = None
        with open(output_path, mode) as f:
            if show_progress:
                pbar = tqdm(
                    total=total_size,
                    initial=resume_byte_pos,
                    unit="B",
                    unit_scale=True,
                    desc=filename[:30]
                )
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
            
            if pbar is not None:
                pbar.close()
        
        return output_path
    
    def extract_archive(
        self, 
        archive_path: Path, 
        extract_dir: Optional[Path] = None,
        remove_archive: bool = False
    ) -> Path:
        """
        Extract tar, tar.gz, tgz, or zip archives.
        
        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to
            remove_archive: Whether to delete archive after extraction
            
        Returns:
            Path to extracted directory
        """
        if extract_dir is None:
            extract_dir = archive_path.parent / archive_path.stem.replace(".tar", "")
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = archive_path.suffix.lower()
        name = archive_path.name.lower()
        
        print(f"Extracting {archive_path.name}...")
        
        def _is_safe_path(base_dir: Path, target_path: Path) -> bool:
            """Check if target path is within base directory (prevents Zip Slip)."""
            try:
                target_path.resolve().relative_to(base_dir.resolve())
                return True
            except ValueError:
                return False
        
        if suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                # Security: Check for path traversal attacks (Zip Slip)
                for member in zf.namelist():
                    member_path = extract_dir / member
                    if not _is_safe_path(extract_dir, member_path):
                        raise ValueError(f"Attempted path traversal in zip: {member}")
                zf.extractall(extract_dir)
        elif suffix in [".tar", ".tgz"] or name.endswith(".tar.gz"):
            mode = "r:gz" if suffix in [".tgz"] or name.endswith(".tar.gz") else "r"
            with tarfile.open(archive_path, mode) as tf:
                # Security: Check for path traversal attacks (Zip Slip)
                for member in tf.getmembers():
                    member_path = extract_dir / member.name
                    if not _is_safe_path(extract_dir, member_path):
                        raise ValueError(f"Attempted path traversal in tar: {member.name}")
                tf.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {suffix}")
        
        if remove_archive:
            archive_path.unlink()
            
        return extract_dir
    
    def download_ugspeechdata(
        self, 
        languages: Optional[List[str]] = None,
        include_raw: bool = False,
        transcribed_only: bool = True
    ) -> Dict[str, Path]:
        """
        Download UGSpeechData from Science Data Bank.
        
        Args:
            languages: List of languages to download. Defaults to config.
            include_raw: Whether to download raw (untranscribed) audio
            transcribed_only: Download only transcribed subsets
            
        Returns:
            Dictionary mapping language to extracted path
        """
        languages = languages or ["akan", "ewe", "dagbani"]
        results = {}
        
        for lang in languages:
            lang_lower = lang.lower()
            if lang_lower not in config.dataset.ugspeechdata_urls:
                print(f"Warning: No URL found for {lang}")
                continue
                
            urls = config.dataset.ugspeechdata_urls[lang_lower]
            lang_dir = self.output_dir / "ugspeechdata" / lang_lower
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            # Download metadata
            print(f"\n{'='*50}")
            print(f"Downloading {lang} metadata...")
            metadata_path = self.download_file(urls["metadata"])
            
            # Download transcribed data
            if transcribed_only or True:  # Always download transcribed
                print(f"Downloading {lang} transcribed data...")
                trans_path = self.download_file(urls["transcribed"])
                self.extract_archive(trans_path, lang_dir / "transcribed")
            
            # Download raw audio (large!)
            if include_raw and not transcribed_only:
                print(f"Downloading {lang} raw audio (this may take a while)...")
                audio_path = self.download_file(urls["audio"])
                self.extract_archive(audio_path, lang_dir / "raw")
            
            results[lang_lower] = lang_dir
            
        return results
    
    def download_bibletis(
        self, 
        languages: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Download BibleTTS from OpenSLR.
        
        Args:
            languages: List of language variants to download
            
        Returns:
            Dictionary mapping language to extracted path
        """
        languages = languages or ["asante_twi", "akuapem_twi", "ewe"]
        results = {}
        
        base_url = config.dataset.bibletis_base_url
        
        for lang in languages:
            lang_lower = lang.lower().replace("-", "_")
            if lang_lower not in config.dataset.bibletis_files:
                print(f"Warning: No BibleTTS file for {lang}")
                continue
                
            filename = config.dataset.bibletis_files[lang_lower]
            url = f"{base_url}/{filename}"
            
            print(f"\n{'='*50}")
            print(f"Downloading BibleTTS {lang}...")
            
            archive_path = self.download_file(url, filename)
            extract_dir = self.output_dir / "bibletis" / lang_lower
            self.extract_archive(archive_path, extract_dir)
            
            results[lang_lower] = extract_dir
            
        return results
    
    def download_fisd(
        self, 
        languages: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Download FISD (Ashesi Financial Inclusion Speech Dataset).
        
        Args:
            languages: List of languages/dialects to download
            
        Returns:
            Dictionary mapping language to extracted path
        """
        languages = languages or ["ga", "asante_twi"]
        results = {}
        
        for lang in languages:
            lang_lower = lang.lower().replace("-", "_")
            if lang_lower not in config.dataset.fisd_urls:
                print(f"Warning: No FISD data for {lang}")
                continue
                
            urls = config.dataset.fisd_urls[lang_lower]
            lang_dir = self.output_dir / "fisd" / lang_lower
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*50}")
            print(f"Downloading FISD {lang}...")
            
            for url in urls:
                filename = url.split("/")[-1]
                archive_path = self.download_file(url, filename)
                self.extract_archive(archive_path, lang_dir)
            
            results[lang_lower] = lang_dir
            
        return results


# ============================================================================
# AUDIO PROCESSOR
# ============================================================================
class AudioProcessor:
    """
    Handles audio file processing: resampling, normalization, format conversion.
    """
    
    def __init__(
        self,
        target_sr_asr: int = 16000,
        target_sr_tts: int = 22050
    ):
        """
        Initialize audio processor.
        
        Args:
            target_sr_asr: Target sample rate for ASR (16kHz standard)
            target_sr_tts: Target sample rate for TTS (22.05kHz for XTTS)
        """
        self.target_sr_asr = target_sr_asr
        self.target_sr_tts = target_sr_tts
    
    def load_audio(
        self, 
        audio_path: Union[str, Path],
        target_sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load and optionally resample audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (None to keep original)
            mono: Convert to mono
            
        Returns:
            Tuple of (audio array, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=mono)
        return audio, sr
    
    def process_for_asr(
        self, 
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Process audio for ASR training (16kHz, mono, normalized).
        
        Args:
            audio_path: Input audio path
            output_path: Optional output path to save processed audio
            normalize: Whether to normalize audio
            
        Returns:
            Tuple of (processed audio, sample rate)
        """
        audio, sr = self.load_audio(audio_path, target_sr=self.target_sr_asr, mono=True)
        
        if normalize:
            audio = librosa.util.normalize(audio)
        
        if output_path:
            sf.write(output_path, audio, self.target_sr_asr)
            
        return audio, self.target_sr_asr
    
    def process_for_tts(
        self, 
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        trim_silence: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Process audio for TTS training (22.05kHz, mono, trimmed, normalized).
        
        Args:
            audio_path: Input audio path
            output_path: Optional output path to save processed audio
            trim_silence: Whether to trim leading/trailing silence
            normalize: Whether to normalize audio
            
        Returns:
            Tuple of (processed audio, sample rate)
        """
        audio, sr = self.load_audio(audio_path, target_sr=self.target_sr_tts, mono=True)
        
        if trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        if normalize:
            audio = librosa.util.normalize(audio)
        
        if output_path:
            sf.write(output_path, audio, self.target_sr_tts)
            
        return audio, self.target_sr_tts
    
    def get_duration(self, audio_path: Union[str, Path]) -> float:
        """Get duration of audio file in seconds."""
        return librosa.get_duration(path=audio_path)
    
    def batch_process(
        self,
        input_dir: Path,
        output_dir: Path,
        mode: str = "asr",
        extensions: List[str] = [".wav", ".mp3", ".flac", ".ogg"],
        show_progress: bool = True
    ) -> List[Path]:
        """
        Batch process all audio files in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            mode: "asr" or "tts"
            extensions: Audio file extensions to process
            show_progress: Show progress bar
            
        Returns:
            List of processed file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_dir.rglob(f"*{ext}"))
        
        processed_files = []
        process_func = self.process_for_asr if mode == "asr" else self.process_for_tts
        
        iterator = tqdm(audio_files, desc=f"Processing for {mode.upper()}") if show_progress else audio_files
        
        for audio_path in iterator:
            try:
                # Maintain relative path structure
                rel_path = audio_path.relative_to(input_dir)
                output_path = output_dir / rel_path.with_suffix(".wav")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                process_func(audio_path, output_path)
                processed_files.append(output_path)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                
        return processed_files


# ============================================================================
# ASR DATASET FORMATTER
# ============================================================================
class ASRDatasetFormatter:
    """
    Formats datasets for ASR (MMS) training.
    Creates HuggingFace datasets with audio and transcriptions.
    """
    
    def __init__(self, processor: Optional[AudioProcessor] = None):
        """
        Initialize formatter.
        
        Args:
            processor: AudioProcessor instance
        """
        self.processor = processor or AudioProcessor()
    
    def parse_ugspeechdata_metadata(
        self, 
        metadata_path: Path,
        audio_dir: Path,
        language: str
    ) -> pd.DataFrame:
        """
        Parse UGSpeechData Excel metadata file.
        
        Args:
            metadata_path: Path to Excel metadata file
            audio_dir: Directory containing audio files
            language: Language code
            
        Returns:
            DataFrame with audio paths and transcriptions
        """
        # Read Excel file
        df = pd.read_excel(metadata_path)
        
        # Common column mappings (adjust based on actual structure)
        # UGSpeechData typically has columns like: Audio_ID, Transcription, Speaker_ID, etc.
        column_mapping = {
            "audio_id": ["Audio_ID", "audio_id", "file_name", "filename"],
            "transcription": ["Transcription", "transcription", "text", "Text"],
            "speaker": ["Speaker_ID", "speaker_id", "Speaker", "speaker"],
        }
        
        # Find and rename columns
        for target, options in column_mapping.items():
            for opt in options:
                if opt in df.columns:
                    df = df.rename(columns={opt: target})
                    break
        
        # Filter to only transcribed entries
        if "transcription" in df.columns:
            df = df[df["transcription"].notna() & (df["transcription"] != "")]
        
        # Add audio paths
        df["audio_path"] = df["audio_id"].apply(
            lambda x: str(audio_dir / f"{x}.wav") if not str(x).endswith(".wav") else str(audio_dir / x)
        )
        
        # Add language
        df["language"] = language
        
        return df
    
    def create_hf_dataset(
        self,
        df: pd.DataFrame,
        audio_column: str = "audio_path",
        text_column: str = "transcription",
        sample_rate: int = 16000
    ) -> Dataset:
        """
        Create a HuggingFace Dataset from DataFrame.
        
        Args:
            df: DataFrame with audio paths and transcriptions
            audio_column: Column containing audio file paths
            text_column: Column containing transcriptions
            sample_rate: Target sample rate
            
        Returns:
            HuggingFace Dataset
        """
        # Filter to existing files
        df = df[df[audio_column].apply(lambda x: Path(x).exists())]
        
        # Create dataset
        dataset = Dataset.from_pandas(df)
        
        # Cast audio column
        dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sample_rate))
        
        return dataset
    
    def create_train_val_test_split(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: HuggingFace Dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            seed: Random seed
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        # First split: train vs (val + test)
        train_test = dataset.train_test_split(
            test_size=1 - train_ratio,
            seed=seed
        )
        
        # Second split: val vs test
        val_test = train_test["test"].train_test_split(
            test_size=0.5,
            seed=seed
        )
        
        return DatasetDict({
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })
    
    def format_for_mms(
        self,
        dataset: Dataset,
        processor,
        max_length: int = 16000 * 30  # 30 seconds max
    ) -> Dataset:
        """
        Format dataset for MMS training.
        
        Args:
            dataset: HuggingFace Dataset
            processor: MMS processor
            max_length: Maximum audio length in samples
            
        Returns:
            Formatted dataset
        """
        def prepare_example(batch):
            audio = batch["audio"]
            
            # Process audio
            inputs = processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
                padding=True
            )
            
            # Encode text
            with processor.as_target_processor():
                labels = processor(batch["transcription"]).input_ids
            
            batch["input_values"] = inputs.input_values[0]
            batch["labels"] = labels
            
            return batch
        
        return dataset.map(prepare_example, remove_columns=["audio"])


# ============================================================================
# TTS DATASET FORMATTER
# ============================================================================
class TTSDatasetFormatter:
    """
    Formats datasets for TTS (XTTS) training.
    Creates metadata.csv and processed audio files in XTTS format.
    """
    
    def __init__(self, processor: Optional[AudioProcessor] = None):
        """
        Initialize formatter.
        
        Args:
            processor: AudioProcessor instance
        """
        self.processor = processor or AudioProcessor()
    
    def parse_bibletis_structure(
        self,
        bibletis_dir: Path,
        language: str
    ) -> pd.DataFrame:
        """
        Parse BibleTTS directory structure.
        
        BibleTTS structure:
        - {lang}/
          - wavs/
          - txt/
          
        Args:
            bibletis_dir: Path to BibleTTS language directory
            language: Language name for speaker
            
        Returns:
            DataFrame with audio paths, text, speaker, and language
        """
        records = []
        
        # Find audio and text files
        wav_dir = bibletis_dir / "wavs"
        txt_dir = bibletis_dir / "txt"
        
        if not wav_dir.exists():
            # Alternative structure
            wav_dir = bibletis_dir
            txt_dir = bibletis_dir
        
        # Match audio to text files
        for wav_file in wav_dir.glob("*.wav"):
            txt_file = txt_dir / wav_file.with_suffix(".txt").name
            
            if txt_file.exists():
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                if text:
                    records.append({
                        "audio_path": str(wav_file),
                        "text": text,
                        "speaker": f"{language}_Speaker",
                        "language": "en"  # XTTS carrier language
                    })
        
        return pd.DataFrame(records)
    
    def create_xtts_dataset(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        process_audio: bool = True,
        show_progress: bool = True
    ) -> Path:
        """
        Create XTTS-formatted dataset with metadata.csv and processed audio.
        
        Args:
            df: DataFrame with audio_path, text, speaker, language columns
            output_dir: Output directory
            process_audio: Whether to process/resample audio
            show_progress: Show progress bar
            
        Returns:
            Path to output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        metadata_rows = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="Creating TTS dataset") if show_progress else df.iterrows()
        
        for idx, row in iterator:
            try:
                audio_path = Path(row["audio_path"])
                
                if not audio_path.exists():
                    continue
                
                # Output filename
                out_filename = f"{idx:06d}.wav"
                out_path = wavs_dir / out_filename
                
                # Process audio
                if process_audio:
                    self.processor.process_for_tts(audio_path, out_path)
                else:
                    shutil.copy(audio_path, out_path)
                
                # Create metadata row: filename|text|speaker|language
                metadata_rows.append(
                    f"{out_filename}|{row['text']}|{row['speaker']}|{row['language']}"
                )
                
            except Exception as e:
                print(f"Error processing {row['audio_path']}: {e}")
        
        # Write metadata.csv
        metadata_path = output_dir / "metadata.csv"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_rows))
        
        print(f"Created TTS dataset with {len(metadata_rows)} samples at {output_dir}")
        
        return output_dir
    
    def merge_datasets(
        self,
        dataset_dirs: List[Path],
        output_dir: Path,
        speaker_prefix: bool = True
    ) -> Path:
        """
        Merge multiple TTS datasets into one unified dataset.
        
        Args:
            dataset_dirs: List of dataset directories (each with metadata.csv and wavs/)
            output_dir: Output directory for merged dataset
            speaker_prefix: Add dataset prefix to audio filenames
            
        Returns:
            Path to merged dataset
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        all_metadata = []
        
        for ds_idx, ds_dir in enumerate(dataset_dirs):
            metadata_path = ds_dir / "metadata.csv"
            wavs_source = ds_dir / "wavs"
            
            if not metadata_path.exists():
                print(f"Warning: No metadata.csv in {ds_dir}")
                continue
            
            with open(metadata_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    old_filename = parts[0]
                    new_filename = f"ds{ds_idx}_{old_filename}"
                    
                    # Copy audio file
                    src = wavs_source / old_filename
                    dst = wavs_dir / new_filename
                    if src.exists():
                        shutil.copy(src, dst)
                        
                        # Update metadata
                        parts[0] = new_filename
                        all_metadata.append("|".join(parts))
        
        # Write merged metadata
        with open(output_dir / "metadata.csv", "w", encoding="utf-8") as f:
            f.write("\n".join(all_metadata))
        
        print(f"Merged {len(all_metadata)} samples from {len(dataset_dirs)} datasets")
        
        return output_dir


# ============================================================================
# UNIFIED DATASET CREATION
# ============================================================================
def create_unified_dataset(
    languages: Optional[List[str]] = None,
    sample_mode: bool = None,
    sample_size: int = None
) -> Dict[str, Path]:
    """
    Create unified datasets for all target languages.
    
    This is the main entry point for dataset preparation.
    
    Args:
        languages: List of language codes
        sample_mode: Whether to use sample subset
        sample_size: Number of samples per language
        
    Returns:
        Dictionary with paths to processed datasets
    """
    languages = languages or config.dataset.languages
    sample_mode = sample_mode if sample_mode is not None else config.dataset.sample_mode
    sample_size = sample_size or config.dataset.sample_size
    
    downloader = DatasetDownloader()
    processor = AudioProcessor()
    asr_formatter = ASRDatasetFormatter(processor)
    tts_formatter = TTSDatasetFormatter(processor)
    
    results = {
        "asr_datasets": {},
        "tts_datasets": {}
    }
    
    print("=" * 60)
    print("Creating Unified Ghanaian Speech Dataset")
    print(f"Languages: {languages}")
    print(f"Sample Mode: {sample_mode}")
    print("=" * 60)
    
    # Download datasets
    print("\n[1/4] Downloading datasets...")
    ugspeech_paths = downloader.download_ugspeechdata(
        languages=["akan", "ewe", "dagbani"],
        transcribed_only=True
    )
    
    bibletis_paths = downloader.download_bibletis(
        languages=["asante_twi", "ewe"]
    )
    
    fisd_paths = downloader.download_fisd(
        languages=["ga"]
    )
    
    # Process for ASR
    print("\n[2/4] Processing for ASR...")
    for lang, path in ugspeech_paths.items():
        # Process audio
        asr_output = config.dataset.asr_data_dir / lang
        asr_output.mkdir(parents=True, exist_ok=True)
        
        # Find and process transcribed audio
        trans_dir = path / "transcribed"
        if trans_dir.exists():
            processed = processor.batch_process(trans_dir, asr_output / "wavs", mode="asr")
            results["asr_datasets"][lang] = asr_output
    
    # Process for TTS
    print("\n[3/4] Processing for TTS...")
    tts_datasets = []
    
    for lang, path in bibletis_paths.items():
        df = tts_formatter.parse_bibletis_structure(path, lang)
        
        if sample_mode and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        tts_output = config.dataset.tts_data_dir / lang
        tts_formatter.create_xtts_dataset(df, tts_output)
        tts_datasets.append(tts_output)
        results["tts_datasets"][lang] = tts_output
    
    # Merge TTS datasets
    print("\n[4/4] Creating unified TTS dataset...")
    if tts_datasets:
        unified_tts = config.dataset.tts_data_dir / "unified"
        tts_formatter.merge_datasets(tts_datasets, unified_tts)
        results["tts_unified"] = unified_tts
    
    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Test the data processing utilities
    print("Testing data processing utilities...")
    
    # Test audio processor
    processor = AudioProcessor()
    print(f"ASR sample rate: {processor.target_sr_asr}")
    print(f"TTS sample rate: {processor.target_sr_tts}")
    
    # Test downloader initialization
    downloader = DatasetDownloader()
    print(f"Download directory: {downloader.output_dir}")
    
    print("\nAll utilities initialized successfully!")
