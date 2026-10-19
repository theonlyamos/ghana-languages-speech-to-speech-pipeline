#!/usr/bin/env python
"""
train_tts_f5.py — fine-tune F5-TTS (v1 Base) into a natural Twi (or Ewe) voice.

Two steps:

  prepare  BibleTTS folder -> F5 dataset (raw.arrow, duration.json, vocab.txt)
           plus a reference clip (ref.wav / ref.txt) used at inference time.
  train    Fine-tune the pretrained F5TTS_v1_Base checkpoint on that dataset.

The pretrained F5 vocab already contains the open e/o and eng letters used in
Twi/Ewe. Any other character in your data (e.g. tone-marked vowels) is appended to the vocab and the text embedding is
grown to match, so no symbol is silently mapped to "unknown".

Examples:
  python scripts/train_tts_f5.py prepare --input data/raw/bibletis/asante_twi --name twi
  python scripts/train_tts_f5.py train --name twi --epochs 20
  python scripts/infer.py --task speak --text "Maakye, wo ho te sen?" --output twi.wav

Needs: pip install f5-tts   (and a CUDA GPU with ~24 GB for training)
"""
import argparse
import json
import shutil
import sys
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


def find_pairs(root: Path):
    """BibleTTS ships one audio file + one .txt per verse; match them by stem."""
    for audio in sorted(p for ext in ("*.flac", "*.wav") for p in root.rglob(ext)):
        txt = audio.with_suffix(".txt")
        if txt.exists():
            text = normalize(txt.read_text(encoding="utf-8"))
            if text:
                yield audio, text


def missing_symbols(texts, vocab) -> list:
    have = set(vocab)
    return sorted({c for t in texts for c in t} - have)


def cmd_prepare(args):
    import soundfile as sf
    from f5_tts.train.datasets.prepare_csv_wavs import prepare_csv_wavs_dir, save_prepped_dataset

    out_dir = Path(args.out or config.dataset.tts_data_dir / f"f5_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for audio, text in find_pairs(Path(args.input)):
        dur = sf.info(str(audio)).duration
        # F5 trains best on 1–25 s clips; longer ones blow up memory.
        if args.min_sec <= dur <= args.max_sec:
            rows.append((audio.resolve().as_posix(), text, dur))
        if args.limit and len(rows) >= args.limit:
            break
    if not rows:
        sys.exit(f"error: no audio/.txt pairs between {args.min_sec}-{args.max_sec}s under {args.input}")

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

    # Reference clip for inference: the clip closest to 8 s (F5 likes 5–10 s prompts).
    ref_audio, ref_text, _ = min(rows, key=lambda r: abs(r[2] - 8))
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
    for name in ("vocab.txt", "ref.wav", "ref.txt"):
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
    pp.add_argument("--input", required=True, help="Extracted BibleTTS language folder")
    pp.add_argument("--name", default="twi", help="Dataset/voice name (default: twi)")
    pp.add_argument("--out", help="Output dir (default: data/processed/tts/f5_<name>)")
    pp.add_argument("--min-sec", type=float, default=1.0)
    pp.add_argument("--max-sec", type=float, default=25.0)
    pp.add_argument("--limit", type=int, default=0, help="Max clips (0 = all); use ~500 for a smoke run")

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
