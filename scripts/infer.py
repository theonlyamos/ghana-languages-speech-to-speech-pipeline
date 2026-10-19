#!/usr/bin/env python
"""
infer.py — one-shot command-line inference for the Ghana S2S pipeline.

Loads your fine-tuned checkpoints (from ``config.asr.output_dir`` /
``config.tts.output_dir`` by default — where training writes them) and runs a
single task end to end. Falls back to the base MMS / NLLB / XTTS models when no
trained checkpoint is found, so it works before you have trained anything too.

Tasks:
  detect      Identify the spoken language only (MMS-LID).
  transcribe  Speech -> text (ASR), optionally auto-detecting the language.
  translate   Text -> text (no audio); requires --text.
  speak       Text -> speech (TTS only); requires --text.
  s2s         Full pipeline: transcribe -> translate -> synthesize (default).

Examples:
  # Full speech-to-speech: detect language, transcribe, translate, synthesize
  python scripts/infer.py audio.wav --target eng --output out.wav

  # Transcribe only, auto-detecting the language
  python scripts/infer.py audio.wav --task transcribe

  # Detect the spoken language only
  python scripts/infer.py audio.wav --task detect

  # Translate text only (no audio)
  python scripts/infer.py --task translate --text "Maakye" --source aka --target eng

  # Speak text with the trained voice (F5-TTS voice in models/tts_f5 if present)
  python scripts/infer.py --task speak --text "Maakye, wo ho te sen?" --output twi.wav

  # Transcribe + translate but skip speech synthesis
  python scripts/infer.py audio.wav --target eng --no-speak

  # Force the base (untrained) models, on CPU, with JSON output
  python scripts/infer.py audio.wav --base --device cpu --no-8bit --json
"""
import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="One-shot inference for the Ghana Speech-to-Speech pipeline.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="?",
                   help="Input audio file (required for detect/transcribe/s2s).")
    p.add_argument("--task", choices=["detect", "transcribe", "translate", "speak", "s2s"],
                   default="s2s", help="Inference task (default: s2s).")
    p.add_argument("-s", "--source", default="auto",
                   help="Source language code, or 'auto' to detect (default: auto).")
    p.add_argument("-t", "--target", default="eng",
                   help="Target language code (default: eng).")
    p.add_argument("--text", help="Input text for --task translate / speak (instead of audio).")
    p.add_argument("-o", "--output", help="Output path for synthesized audio (s2s).")
    p.add_argument("--speaker-ref",
                   help="Reference .wav for voice cloning (s2s synthesis).")
    p.add_argument("--no-speak", action="store_true",
                   help="For --task s2s: skip TTS (transcribe + translate only).")
    p.add_argument("--asr-model-path", help="Override the ASR checkpoint directory.")
    p.add_argument("--tts-model-path", help="Override the TTS checkpoint directory.")
    p.add_argument("--base", action="store_true",
                   help="Force the base models instead of auto-loading trained checkpoints.")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None,
                   help="Compute device (default: cuda if available, else cpu).")
    p.add_argument("--no-8bit", action="store_true",
                   help="Disable 8-bit quantization (needed for CPU / full precision).")
    p.add_argument("--json", action="store_true", help="Emit results as JSON.")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate inputs per task before loading any heavy models.
    if args.task in ("detect", "transcribe", "s2s"):
        if not args.input:
            sys.exit(f"error: --task {args.task} requires an input audio file.")
        if not Path(args.input).is_file():
            sys.exit(f"error: audio file not found: {args.input}")
    if args.task in ("translate", "speak") and not args.text:
        sys.exit(f"error: --task {args.task} requires --text.")

    # Resolve which weights to use (trained checkpoints vs base models).
    from utils.pipeline import GhanaS2SPipeline, resolve_model_paths
    asr_path, tts_path = resolve_model_paths(
        asr_model_path=args.asr_model_path,
        tts_model_path=args.tts_model_path,
        use_base=args.base,
    )
    print(f"[infer] ASR weights: {asr_path or config.asr.model_id + ' (base)'}", file=sys.stderr)
    print(f"[infer] TTS weights: {tts_path or config.tts.model_name + ' (base)'}", file=sys.stderr)

    # Load only the components this task needs (saves memory and load time).
    source_is_auto = args.source.lower() == "auto"
    load = dict(load_lid=False, load_asr=False, load_translation=False, load_tts=False)
    if args.task == "detect":
        load["load_lid"] = True
    elif args.task == "transcribe":
        load.update(load_asr=True, load_lid=source_is_auto)
    elif args.task == "translate":
        load["load_translation"] = True
    elif args.task == "speak":
        load["load_tts"] = True
    elif args.task == "s2s":
        load.update(load_asr=True, load_translation=True,
                    load_tts=not args.no_speak, load_lid=source_is_auto)

    pipeline = GhanaS2SPipeline(
        device=args.device,
        asr_model_path=asr_path,
        tts_model_path=tts_path,
        use_8bit=not args.no_8bit,
        **load,
    )

    result = {"task": args.task}

    if args.task == "detect":
        scores = pipeline.detect_language(args.input, top_k=5, return_all_scores=True)
        result["detected_language"] = max(scores, key=scores.get)
        result["scores"] = scores

    elif args.task == "transcribe":
        tr = pipeline.listen(args.input, language=args.source, return_confidence=True)
        result.update(language=tr.language, text=tr.text,
                      confidence=tr.confidence, duration=tr.duration)

    elif args.task == "translate":
        src = args.source
        if src.lower() == "auto":
            src = config.pipeline.default_source_lang
            print(f"[infer] 'auto' has no meaning for text input; using source '{src}'.",
                  file=sys.stderr)
        tt = pipeline.think(args.text, source_lang=src, target_lang=args.target)
        result.update(source_lang=tt.source_lang, target_lang=tt.target_lang,
                      source_text=tt.source_text, translated_text=tt.translated_text)

    elif args.task == "speak":
        sp = pipeline.speak(args.text, output_path=args.output, speaker_wav=args.speaker_ref)
        result.update(text=sp.text, audio_path=sp.audio_path, duration=round(sp.duration, 2))

    elif args.task == "s2s":
        if args.no_speak:
            tr = pipeline.listen(args.input, language=args.source, return_confidence=True)
            result.update(language=tr.language, text=tr.text, confidence=tr.confidence)
            if args.target and args.target != tr.language:
                tt = pipeline.think(tr.text, source_lang=tr.language, target_lang=args.target)
                result.update(target_lang=tt.target_lang, translated_text=tt.translated_text)
        else:
            s2s = pipeline.run_pipeline(
                audio_input=args.input,
                source_lang=args.source,
                target_lang=args.target,
                speaker_ref=args.speaker_ref,
                translate=True,
                output_path=args.output,
            )
            result.update(
                language=s2s.transcription.language,
                text=s2s.transcription.text,
                translated_text=(s2s.translation.translated_text if s2s.translation else None),
                target_lang=args.target,
                audio_path=s2s.synthesis.audio_path,
                total_latency=round(s2s.total_latency, 2),
            )

    # Report.
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print()
        for key, value in result.items():
            if key == "scores" and isinstance(value, dict):
                value = ", ".join(f"{lang}:{score:.2f}" for lang, score in value.items())
            print(f"{key:>16}: {value}")


if __name__ == "__main__":
    main()
