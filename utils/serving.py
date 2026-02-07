"""
Serving Utilities for Ghana Speech-to-Speech Pipeline
======================================================
Gradio web interface and FastAPI REST endpoints for deployment.
"""

import os
import sys
import io
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Import config and pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config, get_language_code

# Lazy imports for optional dependencies
gradio = None
fastapi = None


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
def create_gradio_interface(pipeline=None, share: bool = False):
    """
    Create a Gradio web interface for the S2S pipeline.
    
    Args:
        pipeline: GhanaS2SPipeline instance (will create if None)
        share: Whether to create a public shareable link
        
    Returns:
        Gradio interface object
    """
    global gradio
    import gradio as gr
    gradio = gr
    
    # Import pipeline if not provided
    if pipeline is None:
        from .pipeline import GhanaS2SPipeline
        pipeline = GhanaS2SPipeline()
    
    # Language options
    LANGUAGES = {
        "Auto-detect": "auto",
        "Akan (Twi)": "aka",
        "Ewe": "ewe",
        "Ga": "gaa",
        "Dagbani": "dag",
        "English": "eng"
    }
    
    # Languages for target (no auto-detect)
    TARGET_LANGUAGES = {
        "Akan (Twi)": "aka",
        "Ewe": "ewe",
        "Ga": "gaa",
        "Dagbani": "dag",
        "English": "eng"
    }
    
    def transcribe_audio(audio_path: str, language: str) -> str:
        """Transcribe audio to text."""
        if audio_path is None:
            return "Please record or upload audio first."
        
        lang_code = LANGUAGES.get(language, "aka")
        result = pipeline.listen(audio_path, language=lang_code)
        
        # If auto-detected, show the detected language
        if lang_code == "auto":
            return f"[Detected: {result.language}] {result.text}"
        return result.text
    
    def translate_text(text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages."""
        if not text:
            return "Please enter text to translate."
        
        src_code = LANGUAGES.get(source_lang, "aka")
        tgt_code = LANGUAGES.get(target_lang, "eng")
        
        result = pipeline.think(text, source_lang=src_code, target_lang=tgt_code)
        return result.translated_text
    
    def synthesize_speech(
        text: str, 
        speaker: str, 
        reference_audio: Optional[str]
    ) -> Optional[str]:
        """Synthesize speech from text."""
        if not text:
            return None
        
        result = pipeline.speak(
            text=text,
            speaker=speaker if speaker else "Twi_Speaker",
            speaker_wav=reference_audio
        )
        return result.audio_path
    
    def full_pipeline(
        audio_input: str,
        source_lang: str,
        target_lang: str,
        reference_audio: Optional[str]
    ) -> Tuple[str, str, Optional[str]]:
        """Run the full S2S pipeline."""
        if audio_input is None:
            return "Please record or upload audio first.", "", None
        
        src_code = LANGUAGES.get(source_lang, "aka")
        tgt_code = TARGET_LANGUAGES.get(target_lang, "eng")
        
        # Determine if translation is needed (can't know for auto until detected)
        translate = (src_code == "auto") or (src_code != tgt_code)
        
        result = pipeline.run_pipeline(
            audio_input=audio_input,
            source_lang=src_code,
            target_lang=tgt_code,
            speaker_ref=reference_audio,
            translate=translate
        )
        
        # Format results - show detected language if auto
        detected_lang = result.transcription.language
        lang_display = source_lang if src_code != "auto" else f"Auto-detected: {detected_lang}"
        transcription = f"[{lang_display}] {result.transcription.text}"
        translation = ""
        if result.translation:
            translation = f"[{target_lang}] {result.translation.translated_text}"
        
        return transcription, translation, result.synthesis.audio_path
    
    # Build Gradio interface
    with gr.Blocks(
        title="Ghana Speech-to-Speech AI",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("""
        # Ghana Speech-to-Speech AI
        
        Unified multilingual speech model for **Akan (Twi)**, **Ewe**, **Ga**, and **Dagbani**.
        
        ---
        """)
        
        with gr.Tabs():
            # Tab 1: Full S2S Pipeline
            with gr.TabItem("Speech-to-Speech"):
                gr.Markdown("### Record or upload audio, get translated speech output")
                
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Input Audio",
                            type="filepath",
                            sources=["microphone", "upload"]
                        )
                        source_lang = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="Auto-detect",
                            label="Source Language"
                        )
                        target_lang = gr.Dropdown(
                            choices=list(TARGET_LANGUAGES.keys()),
                            value="English",
                            label="Target Language"
                        )
                        reference_audio = gr.Audio(
                            label="Speaker Reference (optional, 6 sec)",
                            type="filepath",
                            sources=["upload"]
                        )
                        run_btn = gr.Button("Run Pipeline", variant="primary")
                    
                    with gr.Column():
                        transcription_output = gr.Textbox(
                            label="Transcription",
                            lines=3
                        )
                        translation_output = gr.Textbox(
                            label="Translation",
                            lines=3
                        )
                        audio_output = gr.Audio(
                            label="Synthesized Speech",
                            type="filepath"
                        )
                
                run_btn.click(
                    fn=full_pipeline,
                    inputs=[audio_input, source_lang, target_lang, reference_audio],
                    outputs=[transcription_output, translation_output, audio_output]
                )
            
            # Tab 2: Transcription Only
            with gr.TabItem("Transcribe"):
                gr.Markdown("### Convert speech to text")
                
                with gr.Row():
                    with gr.Column():
                        asr_audio = gr.Audio(
                            label="Audio Input",
                            type="filepath",
                            sources=["microphone", "upload"]
                        )
                        asr_lang = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="Akan (Twi)",
                            label="Language"
                        )
                        asr_btn = gr.Button("Transcribe", variant="primary")
                    
                    with gr.Column():
                        asr_output = gr.Textbox(
                            label="Transcription",
                            lines=5
                        )
                
                asr_btn.click(
                    fn=transcribe_audio,
                    inputs=[asr_audio, asr_lang],
                    outputs=asr_output
                )
            
            # Tab 3: Translation Only
            with gr.TabItem("Translate"):
                gr.Markdown("### Translate text between languages")
                
                with gr.Row():
                    with gr.Column():
                        mt_input = gr.Textbox(
                            label="Input Text",
                            lines=3,
                            placeholder="Enter text to translate..."
                        )
                        mt_source = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="Akan (Twi)",
                            label="Source Language"
                        )
                        mt_target = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="English",
                            label="Target Language"
                        )
                        mt_btn = gr.Button("Translate", variant="primary")
                    
                    with gr.Column():
                        mt_output = gr.Textbox(
                            label="Translation",
                            lines=5
                        )
                
                mt_btn.click(
                    fn=translate_text,
                    inputs=[mt_input, mt_source, mt_target],
                    outputs=mt_output
                )
            
            # Tab 4: Speech Synthesis
            with gr.TabItem("Synthesize"):
                gr.Markdown("### Convert text to speech")
                
                with gr.Row():
                    with gr.Column():
                        tts_input = gr.Textbox(
                            label="Text to Speak",
                            lines=3,
                            placeholder="Enter text to synthesize..."
                        )
                        tts_speaker = gr.Dropdown(
                            choices=["Twi_Speaker", "Ewe_Speaker", "Ga_Speaker", "Dagbani_Speaker"],
                            value="Twi_Speaker",
                            label="Speaker"
                        )
                        tts_reference = gr.Audio(
                            label="Voice Reference (optional)",
                            type="filepath",
                            sources=["upload"]
                        )
                        tts_btn = gr.Button("Synthesize", variant="primary")
                    
                    with gr.Column():
                        tts_output = gr.Audio(
                            label="Generated Audio",
                            type="filepath"
                        )
                
                tts_btn.click(
                    fn=synthesize_speech,
                    inputs=[tts_input, tts_speaker, tts_reference],
                    outputs=tts_output
                )
        
        gr.Markdown("""
        ---
        **Supported Languages:** Akan (Twi/Fante), Ewe, Ga, Dagbani, English
        
        **Models:** Meta MMS (ASR), NLLB-200 (Translation), XTTS v2 (TTS)
        """)
    
    return interface


def launch_gradio(pipeline=None, share: bool = False, port: int = 7860):
    """
    Launch the Gradio interface.
    
    Args:
        pipeline: GhanaS2SPipeline instance
        share: Create public link
        port: Local port number
    """
    interface = create_gradio_interface(pipeline, share)
    interface.launch(
        share=share,
        server_port=port,
        server_name="0.0.0.0"
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
def create_fastapi_app(pipeline=None):
    """
    Create a FastAPI application for the S2S pipeline.
    
    Args:
        pipeline: GhanaS2SPipeline instance (will create if None)
        
    Returns:
        FastAPI application instance
    """
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    from typing import Optional
    import uvicorn
    
    # Import pipeline if not provided
    if pipeline is None:
        from .pipeline import GhanaS2SPipeline
        pipeline = GhanaS2SPipeline()
    
    app = FastAPI(
        title="Ghana Speech-to-Speech API",
        description="REST API for multilingual speech processing in Ghanaian languages",
        version="1.0.0"
    )
    
    # Request/Response models
    class TranscribeResponse(BaseModel):
        text: str
        language: str
        duration: Optional[float] = None
    
    class TranslateRequest(BaseModel):
        text: str
        source_lang: str = "aka"
        target_lang: str = "eng"
    
    class TranslateResponse(BaseModel):
        source_text: str
        translated_text: str
        source_lang: str
        target_lang: str
    
    class SynthesizeRequest(BaseModel):
        text: str
        speaker: str = "Twi_Speaker"
        language: str = "en"
    
    class S2SResponse(BaseModel):
        transcription: str
        translation: Optional[str] = None
        audio_path: str
        latency: float
    
    # Constants for validation
    VALID_LANGUAGES = {"auto", "aka", "ewe", "gaa", "dag", "eng"}
    VALID_TARGET_LANGUAGES = {"aka", "ewe", "gaa", "dag", "eng"}  # No auto for target
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Check API health status."""
        return {"status": "healthy", "models_loaded": True}
    
    # Transcribe endpoint
    @app.post("/api/transcribe", response_model=TranscribeResponse)
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form("auto")
    ):
        """
        Transcribe audio to text.
        
        - **audio**: Audio file (WAV, MP3, FLAC)
        - **language**: Language code (auto, aka, ewe, gaa, dag, eng)
                       Use "auto" for automatic language detection
        """
        # Input validation
        if language not in VALID_LANGUAGES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid language '{language}'. Must be one of: {VALID_LANGUAGES}"
            )
        
        try:
            # Read and validate file size
            content = await audio.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            # Validate content type
            if audio.content_type and not audio.content_type.startswith("audio/"):
                raise HTTPException(
                    status_code=400, 
                    detail="File must be an audio file"
                )
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(content)
                temp_path = f.name
            
            # Transcribe
            result = pipeline.listen(temp_path, language=language)
            
            # Cleanup
            os.unlink(temp_path)
            
            return TranscribeResponse(
                text=result.text,
                language=result.language,
                duration=result.duration
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Translate endpoint
    @app.post("/api/translate", response_model=TranslateResponse)
    async def translate(request: TranslateRequest):
        """
        Translate text between languages.
        
        - **text**: Text to translate
        - **source_lang**: Source language code
        - **target_lang**: Target language code
        """
        try:
            result = pipeline.think(
                request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang
            )
            
            return TranslateResponse(
                source_text=result.source_text,
                translated_text=result.translated_text,
                source_lang=result.source_lang,
                target_lang=result.target_lang
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Synthesize endpoint
    @app.post("/api/synthesize")
    async def synthesize(request: SynthesizeRequest):
        """
        Synthesize speech from text.
        
        - **text**: Text to synthesize
        - **speaker**: Speaker name
        - **language**: Language code
        """
        try:
            result = pipeline.speak(
                text=request.text,
                speaker=request.speaker,
                language=request.language
            )
            
            return FileResponse(
                result.audio_path,
                media_type="audio/wav",
                filename="synthesized.wav"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Full S2S endpoint
    @app.post("/api/speech-to-speech", response_model=S2SResponse)
    async def speech_to_speech(
        audio: UploadFile = File(...),
        source_lang: str = Form("auto"),
        target_lang: str = Form("eng"),
        translate: bool = Form(True)
    ):
        """
        Full speech-to-speech pipeline.
        
        - **audio**: Input audio file
        - **source_lang**: Source language code (auto, aka, ewe, gaa, dag, eng)
                          Use "auto" for automatic language detection
        - **target_lang**: Target language code (aka, ewe, gaa, dag, eng)
        - **translate**: Whether to translate (False for same-language)
        """
        # Validate languages
        if source_lang not in VALID_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source_lang '{source_lang}'. Must be one of: {VALID_LANGUAGES}"
            )
        if target_lang not in VALID_TARGET_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target_lang '{target_lang}'. Must be one of: {VALID_TARGET_LANGUAGES}"
            )
        
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                content = await audio.read()
                f.write(content)
                temp_path = f.name
            
            # Run pipeline
            result = pipeline.run_pipeline(
                audio_input=temp_path,
                source_lang=source_lang,
                target_lang=target_lang,
                translate=translate
            )
            
            # Cleanup
            os.unlink(temp_path)
            
            return S2SResponse(
                transcription=result.transcription.text,
                translation=result.translation.translated_text if result.translation else None,
                audio_path=result.synthesis.audio_path,
                latency=result.total_latency
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Get output audio
    @app.get("/api/audio/{filename}")
    async def get_audio(filename: str):
        """Download generated audio file."""
        from config import OUTPUT_DIR
        
        # Security: Sanitize filename to prevent path traversal attacks
        safe_filename = Path(filename).name  # Remove any directory components
        if safe_filename != filename or '..' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        audio_path = OUTPUT_DIR / safe_filename
        
        # Verify path is within OUTPUT_DIR (defense in depth)
        try:
            audio_path.resolve().relative_to(OUTPUT_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            str(audio_path),
            media_type="audio/wav",
            filename=safe_filename
        )
    
    # List supported languages
    @app.get("/api/languages")
    async def get_languages():
        """Get list of supported languages."""
        return {
            "asr": ["aka", "ewe", "gaa", "dag", "eng"],
            "translation": ["aka", "ewe", "gaa", "dag", "eng"],
            "tts": ["Twi_Speaker", "Ewe_Speaker", "Ga_Speaker", "Dagbani_Speaker"]
        }
    
    return app


def run_fastapi(pipeline=None, host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server.
    
    Args:
        pipeline: GhanaS2SPipeline instance
        host: Host address
        port: Port number
    """
    import uvicorn
    
    app = create_fastapi_app(pipeline)
    uvicorn.run(app, host=host, port=port)


# ============================================================================
# DOCKER CONFIGURATION
# ============================================================================
DOCKERFILE_CONTENT = """
# Ghana Speech-to-Speech API Docker Image
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    ffmpeg \\
    libsndfile1 \\
    espeak-ng \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 7860

# Run the API server
CMD ["python3", "-m", "uvicorn", "utils.serving:create_fastapi_app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKER_COMPOSE_CONTENT = """
version: '3.8'

services:
  ghana-s2s-api:
    build: .
    ports:
      - "8000:8000"
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped
"""

REQUIREMENTS_CONTENT = """
# Core dependencies
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
datasets>=2.14.0
peft>=0.7.0
bitsandbytes>=0.41.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.11.0

# TTS
TTS>=0.22.0

# API/Serving
fastapi>=0.104.0
uvicorn>=0.24.0
gradio>=4.0.0
python-multipart>=0.0.6
pydantic>=2.0.0

# Data processing
pandas>=2.0.0
openpyxl>=3.1.0
tqdm>=4.66.0

# Evaluation
jiwer>=3.0.0
evaluate>=0.4.0
sacrebleu>=2.3.0

# Utilities
requests>=2.31.0
aiofiles>=23.0.0
"""


def generate_docker_files(output_dir: Path = None):
    """
    Generate Docker configuration files.
    
    Args:
        output_dir: Directory to save files (defaults to project root)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent
    
    output_dir = Path(output_dir)
    
    # Write Dockerfile
    with open(output_dir / "Dockerfile", "w") as f:
        f.write(DOCKERFILE_CONTENT.strip())
    
    # Write docker-compose.yml
    with open(output_dir / "docker-compose.yml", "w") as f:
        f.write(DOCKER_COMPOSE_CONTENT.strip())
    
    # Write requirements.txt
    with open(output_dir / "requirements.txt", "w") as f:
        f.write(REQUIREMENTS_CONTENT.strip())
    
    print(f"Docker files generated in {output_dir}")
    print("  - Dockerfile")
    print("  - docker-compose.yml")
    print("  - requirements.txt")
    print("\nTo build and run:")
    print("  docker-compose up --build")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def main():
    """Command-line interface for serving."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ghana S2S Pipeline Server")
    parser.add_argument(
        "--mode",
        choices=["gradio", "fastapi", "docker"],
        default="gradio",
        help="Serving mode"
    )
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    
    args = parser.parse_args()
    
    if args.mode == "gradio":
        launch_gradio(share=args.share, port=args.port)
    elif args.mode == "fastapi":
        run_fastapi(host=args.host, port=args.port)
    elif args.mode == "docker":
        generate_docker_files()


if __name__ == "__main__":
    main()
