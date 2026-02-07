"""
Ghana Speech-to-Speech Pipeline
================================
Unified pipeline class that connects ASR, Translation, and TTS components.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

import torch
import numpy as np

# Audio processing
try:
    import librosa
    import soundfile as sf
except ImportError:
    warnings.warn("librosa/soundfile not installed.")

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config, get_language_code, MODEL_DIR, OUTPUT_DIR


# ============================================================================
# PIPELINE COMPONENTS
# ============================================================================
@dataclass
class TranscriptionResult:
    """Result from ASR transcription."""
    text: str
    language: str
    confidence: Optional[float] = None
    duration: Optional[float] = None


@dataclass
class TranslationResult:
    """Result from translation."""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str


@dataclass
class SynthesisResult:
    """Result from TTS synthesis."""
    audio_path: str
    text: str
    speaker: str
    duration: Optional[float] = None


@dataclass
class S2SResult:
    """Full Speech-to-Speech result."""
    transcription: TranscriptionResult
    translation: Optional[TranslationResult]
    synthesis: SynthesisResult
    total_latency: float


# ============================================================================
# GHANA S2S PIPELINE
# ============================================================================
class GhanaS2SPipeline:
    """
    Unified Speech-to-Speech pipeline for Ghanaian languages.
    
    Architecture:
    - The Ear (ASR): Meta MMS - Transcribes speech to text
    - The Brain (MT): NLLB-200 - Translates between languages
    - The Mouth (TTS): XTTS v2 - Synthesizes speech from text
    
    Supports: Akan (Twi), Ewe, Ga, Dagbani
    """
    
    # Supported languages for auto-detection (subset of MMS-LID languages)
    SUPPORTED_LANGUAGES = {
        "aka": "Akan (Twi/Fante)",
        "ewe": "Ewe", 
        "gaa": "Ga",
        "dag": "Dagbani",
        "eng": "English"
    }
    
    def __init__(
        self,
        device: Optional[str] = None,
        asr_model_path: Optional[str] = None,
        tts_model_path: Optional[str] = None,
        load_asr: bool = True,
        load_tts: bool = True,
        load_translation: bool = True,
        load_lid: bool = True,
        use_8bit: bool = True
    ):
        """
        Initialize the S2S pipeline.
        
        Args:
            device: Device to use ("cuda" or "cpu")
            asr_model_path: Path to fine-tuned ASR model (None for base MMS)
            tts_model_path: Path to fine-tuned TTS model (None for base XTTS)
            load_asr: Whether to load ASR model
            load_tts: Whether to load TTS model
            load_translation: Whether to load translation model
            load_lid: Whether to load Language ID model (enables auto-detection)
            use_8bit: Use 8-bit quantization for memory efficiency
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_8bit = use_8bit
        
        print(f"Initializing Ghana S2S Pipeline on {self.device}")
        print("=" * 60)
        
        # Model paths
        self.asr_model_path = asr_model_path
        self.tts_model_path = tts_model_path
        
        # Initialize components
        self.asr_model = None
        self.asr_processor = None
        self.mt_model = None
        self.mt_tokenizer = None
        self.tts = None
        self.lid_model = None
        self.lid_processor = None
        
        # Load models
        if load_lid:
            self._load_lid()
        if load_asr:
            self._load_asr()
        if load_translation:
            self._load_translation()
        if load_tts:
            self._load_tts()
        
        print("=" * 60)
        print("Pipeline ready!")
    
    def _load_lid(self):
        """Load Language Identification (MMS-LID) model."""
        from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
        
        print("Loading Language ID Model (The Ears' Tuner)...")
        
        model_id = "facebook/mms-lid-4017"
        
        self.lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.lid_processor = AutoFeatureExtractor.from_pretrained(model_id)
        
        # Build label mapping from model config
        self.lid_labels = self.lid_model.config.id2label
        
        print(f"  Loaded: {model_id}")
        print(f"  Supports {len(self.lid_labels)} languages including Ghanaian languages")
    
    def _load_asr(self):
        """Load ASR (MMS) model."""
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        
        print("Loading ASR Model (The Ear)...")
        
        model_id = self.asr_model_path or config.asr.model_id
        
        if self.use_8bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.asr_model = Wav2Vec2ForCTC.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.asr_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        
        self.asr_processor = AutoProcessor.from_pretrained(model_id)
        
        # Set default language
        self._current_asr_lang = None
        
        print(f"  Loaded: {model_id}")
    
    def _load_translation(self):
        """Load Translation (NLLB) model."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("Loading Translation Model (The Brain)...")
        
        model_id = config.translation.model_id
        
        self.mt_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        
        print(f"  Loaded: {model_id}")
    
    def _load_tts(self):
        """Load TTS (XTTS) model."""
        try:
            from TTS.api import TTS
            
            print("Loading TTS Model (The Mouth)...")
            
            if self.tts_model_path:
                # Load fine-tuned model
                config_path = Path(self.tts_model_path) / "config.json"
                model_path = Path(self.tts_model_path) / "best_model.pth"
                
                if model_path.exists():
                    self.tts = TTS(
                        model_path=str(model_path),
                        config_path=str(config_path),
                        progress_bar=False
                    ).to(self.device)
                else:
                    print(f"  Warning: Model not found at {model_path}, using base model")
                    self.tts = TTS(config.tts.model_name, progress_bar=False).to(self.device)
            else:
                # Load base XTTS v2
                self.tts = TTS(config.tts.model_name, progress_bar=False).to(self.device)
            
            print(f"  Loaded TTS model")
            
        except ImportError:
            warnings.warn("Coqui TTS not installed. TTS functionality will be disabled.")
            self.tts = None
    
    def _set_asr_language(self, lang_code: str):
        """
        Set the ASR language adapter.
        
        Args:
            lang_code: ISO 639-3 language code (aka, ewe, gaa, dag, eng)
        """
        if self._current_asr_lang != lang_code:
            self.asr_processor.tokenizer.set_target_lang(lang_code)
            self.asr_model.load_adapter(lang_code)
            self._current_asr_lang = lang_code
    
    # ========================================================================
    # LANGUAGE DETECTION
    # ========================================================================
    
    def detect_language(
        self,
        audio_input: Union[str, Path, np.ndarray],
        top_k: int = 5,
        return_all_scores: bool = False
    ) -> Union[str, Dict[str, float]]:
        """
        Detect the language of speech audio using MMS-LID.
        
        This uses Meta's MMS Language Identification model which supports 4000+ 
        languages including Ghanaian languages (Akan, Ewe, Ga, Dagbani).
        
        Args:
            audio_input: Path to audio file or numpy array of audio samples (16kHz)
            top_k: Number of top languages to return when return_all_scores=True
            return_all_scores: If True, return dict with top-k language scores
                               If False, return only the best matching language code
        
        Returns:
            If return_all_scores=False: Language code string (e.g., "aka", "ewe")
            If return_all_scores=True: Dict mapping language codes to confidence scores
        
        Example:
            >>> pipeline.detect_language("audio.wav")
            "aka"
            >>> pipeline.detect_language("audio.wav", top_k=3, return_all_scores=True)
            {"aka": 0.85, "twi": 0.10, "eng": 0.03}
        """
        if self.lid_model is None:
            raise RuntimeError(
                "Language ID model not loaded. Initialize with load_lid=True"
            )
        
        # Load audio
        if isinstance(audio_input, (str, Path)):
            audio, sr = librosa.load(audio_input, sr=16000, mono=True)
        else:
            audio = audio_input
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono
        
        # Process audio
        inputs = self.lid_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.lid_model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = probs.topk(top_k)
        
        # Map to language codes
        results = {}
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            lang_code = self.lid_labels[idx]
            results[lang_code] = float(prob)
        
        if return_all_scores:
            return results
        else:
            # Return best match, preferring Ghanaian languages
            best_lang = list(results.keys())[0]
            best_score = results[best_lang]
            
            # If best match is a Ghanaian language variant, normalize it
            # MMS-LID may return "twi" which should map to "aka"
            lang_mapping = {
                "twi": "aka",
                "fante": "aka",
                "akan": "aka",
            }
            best_lang = lang_mapping.get(best_lang.lower(), best_lang)
            
            return best_lang
    
    def detect_language_with_fallback(
        self,
        audio_input: Union[str, Path, np.ndarray],
        confidence_threshold: float = 0.3
    ) -> Tuple[str, float, bool]:
        """
        Detect language with fallback to default if confidence is low.
        
        Args:
            audio_input: Path to audio file or numpy array
            confidence_threshold: Minimum confidence to accept detection
        
        Returns:
            Tuple of (language_code, confidence, is_fallback)
        """
        if self.lid_model is None:
            # No LID model, use default
            return (config.pipeline.default_source_lang, 0.0, True)
        
        scores = self.detect_language(audio_input, top_k=5, return_all_scores=True)
        
        best_lang = list(scores.keys())[0]
        best_score = scores[best_lang]
        
        # Map variants to standard codes
        lang_mapping = {"twi": "aka", "fante": "aka", "akan": "aka"}
        best_lang = lang_mapping.get(best_lang.lower(), best_lang)
        
        # Check if it's a supported Ghanaian language
        if best_lang not in self.SUPPORTED_LANGUAGES:
            # Check if any supported language is in top-5
            for lang, score in scores.items():
                mapped = lang_mapping.get(lang.lower(), lang)
                if mapped in self.SUPPORTED_LANGUAGES and score > confidence_threshold:
                    return (mapped, score, False)
            # Fallback to default
            return (config.pipeline.default_source_lang, best_score, True)
        
        if best_score < confidence_threshold:
            return (config.pipeline.default_source_lang, best_score, True)
        
        return (best_lang, best_score, False)
    
    # ========================================================================
    # CORE PIPELINE METHODS
    # ========================================================================
    
    def listen(
        self,
        audio_path: Union[str, Path],
        language: str = "aka",
        return_confidence: bool = False,
        auto_detect_threshold: float = 0.3
    ) -> TranscriptionResult:
        """
        Transcribe speech to text (ASR).
        
        Args:
            audio_path: Path to audio file
            language: Language code (aka, ewe, gaa, dag, eng) or "auto" for 
                      automatic language detection using MMS-LID
            return_confidence: Whether to compute confidence score
            auto_detect_threshold: Confidence threshold for auto-detection
                                   (only used when language="auto")
            
        Returns:
            TranscriptionResult with transcribed text
            
        Example:
            >>> result = pipeline.listen("audio.wav", language="auto")
            >>> print(f"Detected: {result.language}, Text: {result.text}")
        """
        if self.asr_model is None:
            raise RuntimeError("ASR model not loaded. Initialize with load_asr=True")
        
        # Handle automatic language detection
        detected_confidence = None
        if language.lower() == "auto":
            if self.lid_model is None:
                warnings.warn(
                    "Language auto-detection requested but LID model not loaded. "
                    "Falling back to default language. Initialize with load_lid=True "
                    "to enable auto-detection."
                )
                lang_code = get_language_code(
                    config.pipeline.default_source_lang, format="mms"
                )
            else:
                detected_lang, detected_confidence, is_fallback = \
                    self.detect_language_with_fallback(audio_path, auto_detect_threshold)
                lang_code = get_language_code(detected_lang, format="mms")
                if is_fallback:
                    print(f"  [LID] Low confidence ({detected_confidence:.2f}), "
                          f"using fallback: {lang_code}")
                else:
                    print(f"  [LID] Detected language: {lang_code} "
                          f"(confidence: {detected_confidence:.2f})")
        else:
            # Convert language to MMS code
            lang_code = get_language_code(language, format="mms")
        
        # Set language adapter
        self._set_asr_language(lang_code)
        
        # Load and process audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr
        
        # Process
        inputs = self.asr_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.asr_model(**inputs)
            logits = outputs.logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)[0]
        transcription = self.asr_processor.decode(predicted_ids)
        
        # Compute confidence if requested
        confidence = None
        if return_confidence:
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values.mean().item()
        
        return TranscriptionResult(
            text=transcription,
            language=lang_code,
            confidence=confidence,
            duration=duration
        )
    
    def think(
        self,
        text: str,
        source_lang: str = "aka",
        target_lang: str = "eng"
    ) -> TranslationResult:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            TranslationResult with translated text
        """
        if self.mt_model is None:
            raise RuntimeError("Translation model not loaded. Initialize with load_translation=True")
        
        # Convert to NLLB codes
        src_code = get_language_code(source_lang, format="nllb")
        tgt_code = get_language_code(target_lang, format="nllb")
        
        # Tokenize
        self.mt_tokenizer.src_lang = src_code
        inputs = self.mt_tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate translation
        with torch.no_grad():
            translated_tokens = self.mt_model.generate(
                **inputs,
                forced_bos_token_id=self.mt_tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=config.translation.max_length,
                num_beams=config.translation.num_beams
            )
        
        # Decode
        translated_text = self.mt_tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]
        
        return TranslationResult(
            source_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang
        )
    
    def speak(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker: str = "Twi_Speaker",
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en"
    ) -> SynthesisResult:
        """
        Synthesize speech from text (TTS).
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio (None for temp file)
            speaker: Speaker name (for multi-speaker models)
            speaker_wav: Reference audio for voice cloning
            language: Language code for TTS (use "en" for XTTS)
            
        Returns:
            SynthesisResult with audio path
        """
        if self.tts is None:
            raise RuntimeError("TTS model not loaded. Initialize with load_tts=True")
        
        # Create output path if not provided
        if output_path is None:
            # Security: Use mkstemp instead of deprecated mktemp
            import os
            fd, output_path = tempfile.mkstemp(suffix=".wav", dir=str(OUTPUT_DIR))
            os.close(fd)  # Close the file descriptor, let the TTS library write to it
        
        output_path = str(output_path)
        
        # Synthesize
        if speaker_wav:
            # Voice cloning mode
            self.tts.tts_to_file(
                text=text,
                speaker_wav=str(speaker_wav),
                language=language,
                file_path=output_path
            )
        else:
            # Standard speaker mode
            try:
                self.tts.tts_to_file(
                    text=text,
                    speaker=speaker,
                    language=language,
                    file_path=output_path
                )
            except (ValueError, KeyError) as e:
                # Only catch speaker-related errors, not all exceptions
                if "speaker" in str(e).lower():
                    warnings.warn(f"Speaker '{speaker}' not available, using default")
                    self.tts.tts_to_file(
                        text=text,
                        language=language,
                        file_path=output_path
                    )
                else:
                    raise
        
        # Get duration
        duration = librosa.get_duration(path=output_path)
        
        return SynthesisResult(
            audio_path=output_path,
            text=text,
            speaker=speaker,
            duration=duration
        )
    
    def run_pipeline(
        self,
        audio_input: Union[str, Path],
        source_lang: str = "aka",
        target_lang: str = "eng",
        speaker_ref: Optional[Union[str, Path]] = None,
        translate: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        auto_detect_threshold: float = 0.3
    ) -> S2SResult:
        """
        Run the full Speech-to-Speech pipeline.
        
        Args:
            audio_input: Path to input audio file
            source_lang: Source language code, or "auto" for automatic detection
            target_lang: Target language code
            speaker_ref: Reference audio for voice cloning
            translate: Whether to translate (False for same-language S2S)
            output_path: Path to save output audio
            auto_detect_threshold: Confidence threshold for auto-detection
            
        Returns:
            S2SResult with all intermediate and final results
            
        Example:
            >>> # Auto-detect input language, translate to English
            >>> result = pipeline.run_pipeline("twi_audio.wav", source_lang="auto", target_lang="eng")
        """
        start_time = time.time()
        
        # Handle auto-detection for source language
        detected_lang = source_lang
        if source_lang.lower() == "auto":
            if self.lid_model is not None:
                detected_lang, confidence, is_fallback = \
                    self.detect_language_with_fallback(audio_input, auto_detect_threshold)
                if is_fallback:
                    print(f"[LID] Low confidence ({confidence:.2f}), using: {detected_lang}")
                else:
                    print(f"[LID] Detected language: {detected_lang} (confidence: {confidence:.2f})")
            else:
                detected_lang = config.pipeline.default_source_lang
                print(f"[LID] Model not loaded, using default: {detected_lang}")
        
        print(f"\n{'='*50}")
        print(f"Ghana S2S Pipeline: {detected_lang.upper()} -> {target_lang.upper()}")
        print(f"{'='*50}")
        
        # Step 1: ASR (Listen) - use detected language, not "auto"
        print("\n[1/3] Listening...")
        transcription = self.listen(audio_input, language=detected_lang)
        print(f"      Heard ({detected_lang}): {transcription.text}")
        
        # Step 2: Translation (Think)
        translation = None
        text_for_tts = transcription.text
        
        if translate and detected_lang != target_lang:
            print("\n[2/3] Translating...")
            translation = self.think(
                transcription.text,
                source_lang=detected_lang,
                target_lang=target_lang
            )
            text_for_tts = translation.translated_text
            print(f"      Translated ({target_lang}): {text_for_tts}")
        else:
            print("\n[2/3] Skipping translation (same language)")
        
        # Step 3: TTS (Speak)
        print("\n[3/3] Speaking...")
        
        # Determine speaker based on target language
        speaker = config.tts.speakers.get(
            get_language_code(target_lang, "mms"),
            "Twi_Speaker"
        )
        
        synthesis = self.speak(
            text=text_for_tts,
            output_path=output_path,
            speaker=speaker,
            speaker_wav=speaker_ref,
            language=config.tts.carrier_language
        )
        print(f"      Audio saved to: {synthesis.audio_path}")
        
        total_latency = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"Total Latency: {total_latency:.2f}s")
        print(f"{'='*50}")
        
        return S2SResult(
            transcription=transcription,
            translation=translation,
            synthesis=synthesis,
            total_latency=total_latency
        )
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def batch_transcribe(
        self,
        audio_files: List[Union[str, Path]],
        language: str = "aka",
        show_progress: bool = True
    ) -> List[TranscriptionResult]:
        """
        Batch transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language: Language code
            show_progress: Show progress bar
            
        Returns:
            List of TranscriptionResults
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(audio_files, desc="Transcribing") if show_progress else audio_files
        
        for audio_path in iterator:
            try:
                result = self.listen(audio_path, language=language)
                results.append(result)
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")
                results.append(TranscriptionResult(text="", language=language))
        
        return results
    
    def batch_synthesize(
        self,
        texts: List[str],
        output_dir: Union[str, Path],
        speaker: str = "Twi_Speaker",
        speaker_wav: Optional[Union[str, Path]] = None,
        show_progress: bool = True
    ) -> List[SynthesisResult]:
        """
        Batch synthesize multiple texts.
        
        Args:
            texts: List of texts to synthesize
            output_dir: Directory to save audio files
            speaker: Speaker name
            speaker_wav: Reference audio for voice cloning
            show_progress: Show progress bar
            
        Returns:
            List of SynthesisResults
        """
        from tqdm import tqdm
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        iterator = tqdm(enumerate(texts), total=len(texts), desc="Synthesizing") if show_progress else enumerate(texts)
        
        for i, text in iterator:
            try:
                output_path = output_dir / f"audio_{i:04d}.wav"
                result = self.speak(
                    text=text,
                    output_path=output_path,
                    speaker=speaker,
                    speaker_wav=speaker_wav
                )
                results.append(result)
            except Exception as e:
                print(f"Error synthesizing '{text[:30]}...': {e}")
        
        return results
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported languages for each component."""
        return {
            "asr": ["aka", "ewe", "gaa", "dag", "eng"],
            "translation": list(config.translation.lang_codes.keys()),
            "tts": list(config.tts.speakers.keys())
        }
    
    def check_language_support(self, lang: str) -> Dict[str, bool]:
        """Check if a language is supported by each component."""
        lang_code = get_language_code(lang, "mms")
        supported = self.get_supported_languages()
        
        return {
            "asr": lang_code in supported["asr"],
            "translation": lang in supported["translation"] or lang_code in supported["translation"],
            "tts": lang_code in supported["tts"]
        }
    
    def unload_models(self):
        """Unload all models to free memory."""
        if self.asr_model is not None:
            del self.asr_model
            self.asr_model = None
        
        if self.mt_model is not None:
            del self.mt_model
            self.mt_model = None
        
        if self.tts is not None:
            del self.tts
            self.tts = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("All models unloaded.")


# ============================================================================
# STREAMING PIPELINE (FOR REAL-TIME APPLICATIONS)
# ============================================================================
class StreamingS2SPipeline(GhanaS2SPipeline):
    """
    Streaming version of the S2S pipeline for real-time applications.
    Supports chunked audio processing and WebSocket communication.
    """
    
    def __init__(self, *args, chunk_size: float = 3.0, max_buffer_duration: float = 30.0, **kwargs):
        """
        Initialize streaming pipeline.
        
        Args:
            chunk_size: Audio chunk size in seconds
            max_buffer_duration: Maximum buffer duration before forced truncation (safety limit)
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.max_buffer_duration = max_buffer_duration
        self.audio_buffer = []
    
    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        language: str = "aka"
    ) -> Optional[str]:
        """
        Process a single audio chunk.
        
        Args:
            audio_chunk: Audio data as numpy array
            sample_rate: Sample rate of audio
            language: Language code
            
        Returns:
            Partial transcription or None if buffer not full
        """
        self.audio_buffer.append(audio_chunk)
        
        # Check if we have enough audio
        total_samples = sum(len(c) for c in self.audio_buffer)
        total_duration = total_samples / sample_rate
        
        # Safety: Prevent unbounded buffer growth
        if total_duration > self.max_buffer_duration:
            # Keep only the most recent chunk_size worth of audio
            keep_samples = int(sample_rate * self.chunk_size)
            full_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = [full_audio[-keep_samples:]]
            warnings.warn(f"Audio buffer exceeded {self.max_buffer_duration}s, truncating to prevent memory exhaustion")
            total_duration = self.chunk_size
        
        if total_duration >= self.chunk_size:
            # Concatenate and process
            full_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            
            # Save to temp file and transcribe
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, full_audio, sample_rate)
                result = self.listen(f.name, language=language)
                os.unlink(f.name)
            
            return result.text
        
        return None
    
    def reset_buffer(self):
        """Clear the audio buffer."""
        self.audio_buffer = []


# ============================================================================
# QUICK TEST
# ============================================================================
if __name__ == "__main__":
    print("Testing GhanaS2SPipeline initialization...")
    
    # Test with minimal loading
    try:
        pipeline = GhanaS2SPipeline(
            load_asr=False,
            load_tts=False,
            load_translation=False
        )
        print("Pipeline initialized successfully (no models loaded)")
        
        # Check supported languages
        print(f"\nSupported languages: {pipeline.get_supported_languages()}")
        
    except Exception as e:
        print(f"Error: {e}")
