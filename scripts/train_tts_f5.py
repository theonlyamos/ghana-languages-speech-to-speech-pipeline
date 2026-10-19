#!/usr/bin/env python
"""
train_tts_f5.py — fine-tune F5-TTS (v1 Base) into a natural Twi (or Ewe) voice.

Two steps:

  prepare  Audio/text folders (e.g. BibleTTS) and/or Hugging Face datasets
           (WAXAL, Loud Lounge, Twi Multi-Speaker; see HF_SOURCES) -> one F5
           dataset (raw.arrow, duration.json, vocab.txt), a reference clip
           (ref.wav / ref.txt) and sources.json listing each source's license.
  train    Fine-tune the pretrained F5TTS_v1_Base checkpoint on that dataset.

The pretrained F5 vocab already contains the open e/o and eng letters used in
Twi/Ewe. Any other character in your data (e.g. tone-marked vowels) is appended to the vocab and the text embedding is
grown to match, so no symbol is silently mapped to "unknown".

Examples:
  python scripts/train_tts_f5.py prepare --name twi --input data/raw/bibletis/asante_twi \
      --source waxal_twi --source loud_lounge --source twi_multispeaker
  python scripts/train_tts_f5.py train --name twi --epochs 20
  python scripts/infer.py --task speak --text "Maakye, wo ho te sen?" --output twi.wav

Needs: pip install f5-tts   (and a CUDA GPU with ~24 GB for training)
"""
import argparse
import json
import shutil
import sys
import time
import unicodedata
from importlib.resources import files
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import config  # noqa: E402

PRETRAINED_CKPT = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
EMBED_KEY = "ema_model.transformer.text_embed.text_embed.weight"
MEL = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
           target_sample_rate=24000, mel_spec_type="vocos")
# Same architecture as F5TTS_v1_Base (f5_tts/train/finetune_cli.py).
DIT_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
MODEL_DIR_F5 = Path(config.tts.f5_output_dir)


def pretrained_vocab_path() -> Path:
    # The Emilia vocab the pretrained model was trained with; shipped in the wheel.
    return Path(str(files("f5_tts").joinpath("infer/examples/vocab.txt")))


def read_vocab(path: Path) -> list:
    # One symbol per line; F5 reads line[:-1], so index 0 must be the space.
    with open(path, encoding="utf-8") as f:
        return [line[:-1] for line in f]


def normalize(text: str) -> str:
    # NFC so "e" + combining accent and the precomposed letter are the same symbol.
    return " ".join(unicodedata.normalize("NFC", text).split())


# Hugging Face sources, by name. "keep" filters rows before any audio is decoded.
# Multi-speaker data is fine: F5 is a voice-cloning model, so the output voice
# comes from the reference clip, while extra speakers teach it Twi rhythm/prosody.
HF_SOURCES = {
    # Studio voice actors, phonetically balanced scripts (Google WAXAL, Univ. of Ghana).
    "waxal_twi": dict(repo="google/WaxalNLP", config="twi_tts", text="text",
                      splits=("train", "validation", "test"), license="CC-BY-4.0"),
    "waxal_fante": dict(repo="google/WaxalNLP", config="fat_tts", text="text",
                        splits=("train", "validation", "test"), license="CC-BY-4.0"),
    "waxal_ewe": dict(repo="google/WaxalNLP", config="ewe_tts", text="text",
                      splits=("train", "validation", "test"), license="CC-BY-4.0"),
    # ~65 h conversational podcast Twi/English. Skip lines with digits: numbers
    # are written as numerals, not words, so the text wouldn't match the audio.
    "loud_lounge": dict(repo="ghanaopenai/loud-lounge-twi-tts", config=None, text="text",
                        splits=("train", "validation"), license="CC-BY-NC-4.0",
                        keep=lambda r: not r["has_digits"]),
    # ~90 h of 30 s single-speaker clips with quality scores; keep the clean ones.
    # Filter on DNSMOS, not UTMOS: UTMOS is English-trained and scores this Twi
    # data low across the board (median ~2.0), while DNSMOS overall has median ~3.3.
    "twi_multispeaker": dict(repo="ghanaopenai/twi-multispeaker-tts", config=None,
                             text="transcription", splits=("train",), license="CC-BY-NC-4.0",
                             keep=lambda r: r["spk_count"] == 1 and (r["dnsmos_ovr"] or 0) >= 3.0
                             and (r["speech_ratio"] or 0) >= 0.7),
}
FOLDER_LICENSE = "see source (BibleTTS: CC-BY-SA-4.0)"


def find_pairs(root: Path):
    """BibleTTS ships one audio file + one .txt per verse; match them by stem."""
    for audio in sorted(p for ext in ("*.flac", "*.wav") for p in root.rglob(ext)):
        txt = audio.with_suffix(".txt")
        if txt.exists():
            text = normalize(txt.read_text(encoding="utf-8"))
            if text:
                yield audio, text


def folder_rows(root: Path, args):
    import soundfile as sf

    for audio, text in find_pairs(root):
        dur = sf.info(str(audio)).duration
        if args.min_sec <= dur <= args.max_sec:
            yield audio.resolve().as_posix(), text, dur


def hf_rows(name: str, wav_dir: Path, args):
    """Stream a Hugging Face dataset, writing kept clips as 24 kHz mono wavs."""
    import io

    import soundfile as sf
    import torch
    import torchaudio
    from datasets import Audio, load_dataset

    src = HF_SOURCES[name]
    keep = src.get("keep", lambda r: True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    def stream(split, start):
        # decode=False: read the raw bytes with soundfile, no torchcodec/FFmpeg needed.
        ds = load_dataset(src["repo"], src["config"], split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        return enumerate(ds.skip(start) if start else ds, start)

    def rows_with_retry(split, retries=5):
        # Long streams over the network do drop; resume from the next unread row.
        nxt, tries = 0, 0
        while True:
            try:
                for i, r in stream(split, nxt):
                    nxt = i + 1
                    yield i, r
                return
            except (ConnectionError, OSError) as e:
                tries += 1
                if tries > retries:
                    raise
                print(f"  {name}/{split}: stream dropped ({e}); retry {tries}/{retries} from row {nxt}")
                time.sleep(2 ** tries)

    for split in src["splits"]:
        for i, r in rows_with_retry(split):
            text = normalize(r[src["text"]] or "")
            if not text or not keep(r) or not r["audio"].get("bytes"):
                continue
            data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
            dur = len(data) / sr
            if not args.min_sec <= dur <= args.max_sec:
                continue
            wav = torch.from_numpy(data.mean(axis=1))
            if sr != MEL["target_sample_rate"]:  # F5 trains at 24 kHz; also halves disk vs 48 kHz
                wav = torchaudio.functional.resample(wav, sr, MEL["target_sample_rate"])
            path = wav_dir / f"{name}_{split}_{i:07d}.wav"
            sf.write(path, wav.numpy(), MEL["target_sample_rate"])
            yield path.resolve().as_posix(), text, dur


def missing_symbols(texts, vocab) -> list:
    have = set(vocab)
    return sorted({c for t in texts for c in t} - have)


def cmd_prepare(args):
    import soundfile as sf
    from f5_tts.train.datasets.prepare_csv_wavs import prepare_csv_wavs_dir, save_prepped_dataset

    out_dir = Path(args.out or config.dataset.tts_data_dir / f"f5_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [("folder:" + str(p), lambda p=p: folder_rows(Path(p), args), FOLDER_LICENSE)
               for p in args.input]
    for name in args.source:
        lic = HF_SOURCES[name]["license"]
        if args.commercial_only and "NC" in lic:
            print(f"Skipping {name} ({lic}) because of --commercial-only")
            continue
        sources.append((name, lambda n=name: hf_rows(n, out_dir / "wavs", args), lic))
    if not sources:
        sys.exit("error: give at least one --input folder or --source dataset.")

    rows, summary = [], []
    for name, rows_fn, lic in sources:
        print(f"Reading {name} ...")
        got = []
        for row in rows_fn():
            got.append(row)
            if args.limit and len(got) >= args.limit:  # per source
                break
        rows += got
        summary.append(dict(source=name, license=lic, clips=len(got),
                            hours=round(sum(d for *_, d in got) / 3600, 2)))
        print(f"  {len(got)} clips, {summary[-1]['hours']} h")
    if not rows:
        sys.exit(f"error: no usable clips between {args.min_sec}-{args.max_sec}s in any source.")

    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        # F5's reader splits on "|" with no escaping, so keep it out of the text.
        f.write("audio_file|text\n")
        f.writelines(f"{a}|{t.replace('|', ',')}\n" for a, t, _ in rows)

    result, durations, _ = prepare_csv_wavs_dir(str(csv_path))
    # Empty symbol set: F5 writes vocab.txt without an encoding (breaks on Windows),
    # and we write our own below anyway.
    save_prepped_dataset(out_dir, result, durations, set(), is_finetune=False)

    # Overwrite the dataset-only vocab with pretrained vocab + anything new.
    vocab = read_vocab(pretrained_vocab_path())
    extra = missing_symbols([t for _, t, _ in rows], vocab)
    with open(out_dir / "vocab.txt", "w", encoding="utf-8") as f:
        f.write("".join(s + "\n" for s in vocab + extra))
    print(f"New symbols added to vocab: {extra or 'none'}")

    # Record what the voice was trained on: the strictest license applies to the model.
    (out_dir / "sources.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Reference clip (sets the output voice): from the FIRST source that produced
    # clips, closest to 8 s, so list your cleanest single-speaker source first.
    # (Empty sources add no rows, so that source's clips start at rows[0].)
    first_n = next(s["clips"] for s in summary if s["clips"])
    ref_audio, ref_text, _ = min(rows[:first_n], key=lambda r: abs(r[2] - 8))
    data, sr = sf.read(ref_audio)
    sf.write(out_dir / "ref.wav", data, sr)
    (out_dir / "ref.txt").write_text(ref_text, encoding="utf-8")
    print(f"Prepared {len(rows)} clips ({sum(d for *_, d in rows) / 3600:.1f} h) in {out_dir}")


def expand_embeddings(src: str, dst: Path, n_new: int):
    """Copy the pretrained checkpoint, growing the text embedding by n_new rows."""
    import torch
    from safetensors.torch import load_file, save_file

    sd = load_file(src, device="cpu")
    old = sd[EMBED_KEY]
    sd[EMBED_KEY] = torch.cat([old, torch.randn(n_new, old.size(1)) * old.std()])
    save_file(sd, str(dst))


def cmd_train(args):
    import torch
    from cached_path import cached_path
    from datasets import Dataset
    from f5_tts.model import CFM, DiT, Trainer
    from f5_tts.model.dataset import CustomDataset

    data_dir = Path(args.data or config.dataset.tts_data_dir / f"f5_{args.name}")
    ckpt_dir = Path(args.out or MODEL_DIR_F5 / args.name)
    if not (data_dir / "raw.arrow").exists():
        sys.exit(f"error: {data_dir} has no raw.arrow — run `prepare` first.")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vocab = read_vocab(data_dir / "vocab.txt")
    n_new = len(vocab) - len(read_vocab(pretrained_vocab_path()))

    # Trainer resumes from model_last.pt, else starts from a pretrained_* file here.
    if not any(ckpt_dir.glob("model_*.pt")) and not any(ckpt_dir.glob("pretrained_*")):
        src = str(cached_path(PRETRAINED_CKPT))
        dst = ckpt_dir / "pretrained_model_1250000.safetensors"
        if n_new:
            expand_embeddings(src, dst, n_new)
        else:
            shutil.copy2(src, dst)

    # Ship what inference needs next to the checkpoints.
    for name in ("vocab.txt", "ref.wav", "ref.txt", "sources.json"):
        if (data_dir / name).exists():
            shutil.copy2(data_dir / name, ckpt_dir / name)

    model = CFM(
        transformer=DiT(**DIT_CFG, text_num_embeds=len(vocab), mel_dim=MEL["n_mel_channels"]),
        mel_spec_kwargs=MEL,
        vocab_char_map={c: i for i, c in enumerate(vocab)},
    )
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    trainer = Trainer(
        model, args.epochs, args.lr,
        num_warmup_updates=args.warmup,
        save_per_updates=args.save_every,
        last_per_updates=args.save_every // 4 or 1,
        keep_last_n_checkpoints=2,
        checkpoint_path=str(ckpt_dir),
        batch_size_per_gpu=args.batch_frames,
        batch_size_type="frame",
        max_samples=64,
        grad_accumulation_steps=args.grad_accum,
        logger=args.logger,
        wandb_project="ghana-f5-tts",
        wandb_run_name=args.name,
        accelerate_kwargs={"mixed_precision": "bf16" if bf16 else "no"},
    )
    durations = json.loads((data_dir / "duration.json").read_text(encoding="utf-8"))["duration"]
    dataset = CustomDataset(Dataset.from_file(str(data_dir / "raw.arrow")), durations=durations, **MEL)
    trainer.train(dataset, num_workers=args.workers, resumable_with_seed=666)
    print(f"Done. Checkpoints in {ckpt_dir} (inference loads model_last.pt).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="BibleTTS folder -> F5 dataset")
    pp.add_argument("--input", action="append", default=[],
                    help="Folder of audio + same-name .txt files, e.g. extracted BibleTTS (repeatable)")
    pp.add_argument("--source", action="append", default=[], choices=sorted(HF_SOURCES),
                    help="Hugging Face dataset to stream in (repeatable)")
    pp.add_argument("--commercial-only", action="store_true",
                    help="Drop non-commercial (NC) sources")
    pp.add_argument("--name", default="twi", help="Dataset/voice name (default: twi)")
    pp.add_argument("--out", help="Output dir (default: data/processed/tts/f5_<name>)")
    pp.add_argument("--min-sec", type=float, default=1.0)
    pp.add_argument("--max-sec", type=float, default=30.0)
    pp.add_argument("--limit", type=int, default=0,
                    help="Max clips per source (0 = all); use ~200 for a smoke run")

    tp = sub.add_parser("train", help="Fine-tune F5TTS_v1_Base")
    tp.add_argument("--name", default="twi")
    tp.add_argument("--data", help="Prepared dataset dir (default: data/processed/tts/f5_<name>)")
    tp.add_argument("--out", help="Checkpoint dir (default: models/tts_f5/<name>)")
    tp.add_argument("--epochs", type=int, default=20)
    tp.add_argument("--lr", type=float, default=1e-5)
    tp.add_argument("--warmup", type=int, default=1000, help="Warmup updates")
    tp.add_argument("--batch-frames", type=int, default=4000,
                    help="Mel frames per GPU batch (~4000 fits 24 GB; lower if OOM)")
    tp.add_argument("--grad-accum", type=int, default=2)
    tp.add_argument("--save-every", type=int, default=2000, help="Save a numbered checkpoint every N updates")
    tp.add_argument("--workers", type=int, default=4)
    tp.add_argument("--logger", choices=["wandb", "tensorboard"], default=None)

    args = p.parse_args()
    (cmd_prepare if args.cmd == "prepare" else cmd_train)(args)


if __name__ == "__main__":
    main()
