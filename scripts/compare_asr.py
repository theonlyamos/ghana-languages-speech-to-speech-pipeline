#!/usr/bin/env python3
"""Compare speech recognition models on the same Ghanaian test clips.

Models:
  griot  - Qlerqly/griot-nano-1 (153M Conformer-CTC, greedy decoding)
  mms    - our pipeline's ASR stage: facebook/mms-1b-all with the MMS language
           adapter, or a fine-tuned LoRA adapter dir via --mms-lora

Test data is streamed from Hugging Face so nothing large is downloaded:
  aka, ewe, dag : google/WaxalNLP <lang>_asr, test split
  gaa           : ghananlpcommunity/ga-multispeaker-speech-text-20k, test split

Both models' outputs go through the same text normalizer before WER/CER.

Example:
  python scripts/compare_asr.py --langs aka ewe dag gaa --n 200 --out reports/asr_comparison
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16_000

# lang -> (dataset id, config, split, transcript column candidates)
TEST_SETS = {
    "aka": ("google/WaxalNLP", "aka_asr", "test"),
    "ewe": ("google/WaxalNLP", "ewe_asr", "test"),
    "dag": ("google/WaxalNLP", "dag_asr", "test"),
    "gaa": ("ghananlpcommunity/ga-multispeaker-speech-text-20k", None, "test"),
}
TEXT_COLS = ("transcription", "text", "sentence", "transcript", "corrected_text")
AUDIO_COLS = ("audio", "audio_path", "path")


# --------------------------------------------------------------------------- text
LOOKALIKES = {
    "ε": "ɛ", "Ɛ": "ɛ", "Ε": "ɛ", "є": "ɛ",
    "ↄ": "ɔ", "Ɔ": "ɔ", "Ↄ": "ɔ",
    "Ŋ": "ŋ", "Ɖ": "ɖ", "Ƒ": "ƒ", "Ʋ": "ʋ", "Ɣ": "ɣ", "Ʒ": "ʒ", "Ɲ": "ɲ",
    "’": "'", "‘": "'",
}


def normalize(text: str) -> str:
    """Lowercase, map look-alike letters, drop tone marks and punctuation."""
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(LOOKALIKES.get(c, c) for c in text).lower()
    # Strip combining marks (tones/accents) so diacritic conventions don't dominate.
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- data
def load_test_clips(lang: str, n: int, seed: int = 0):
    """Stream n (audio_16k, reference) pairs for a language."""
    from datasets import Audio, load_dataset

    repo, cfg, split = TEST_SETS[lang]
    ds = load_dataset(repo, cfg, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    cols = list(ds.features.keys()) if ds.features else []
    text_col = next((c for c in TEXT_COLS if c in cols), None)
    audio_col = next((c for c in AUDIO_COLS if c in cols), None)
    if text_col is None or audio_col is None:
        raise SystemExit(f"{repo}/{cfg}: can't find text/audio columns in {cols}")
    ds = ds.cast_column(audio_col, Audio(sampling_rate=SR))

    clips = []
    for ex in ds:
        ref = ex[text_col]
        audio = ex[audio_col]
        if not ref or not str(ref).strip() or audio is None:
            continue
        wav = np.asarray(audio["array"], dtype=np.float32)
        dur = len(wav) / SR
        if dur < 0.5 or dur > 30:
            continue
        clips.append({"lang": lang, "source": f"{repo}:{cfg or ''}:{split}",
                      "ref": str(ref), "wav": wav, "dur": dur})
        if len(clips) >= n:
            break
    return clips


# --------------------------------------------------------------------------- models
class Griot:
    name = "griot-nano-1"

    def __init__(self, model_dir: str | None, device: str):
        from huggingface_hub import snapshot_download

        self.dir = Path(model_dir or snapshot_download("Qlerqly/griot-nano-1"))
        sys.path.insert(0, str(self.dir / "src"))
        sys.path.insert(0, str(self.dir))
        # Reuse the repo's own inference helpers so preprocessing matches training.
        self.inf = importlib.import_module("inference")
        self.device = device
        self.model, self.cfg, self.id2tok = self._load()

    def _load(self):
        import torch
        from safetensors.torch import load_file
        from conformer_ctc.model import ConformerCTC, ConformerCTCConfig

        cfg_raw = json.loads((self.dir / "config.json").read_text())
        model_cfg = cfg_raw.get("model", cfg_raw)
        cfg = (ConformerCTCConfig.from_dict(model_cfg) if hasattr(ConformerCTCConfig, "from_dict")
               else ConformerCTCConfig(**{k: v for k, v in model_cfg.items()
                                          if k in ConformerCTCConfig.__dataclass_fields__}))
        model = ConformerCTC(cfg)
        model.load_state_dict(load_file(str(self.dir / "model.safetensors")))
        model.to(self.device).eval()
        vocab = json.loads((self.dir / "vocab.json").read_text())
        id2tok = {int(v): k for k, v in vocab.items()}
        return model, cfg, id2tok

    def transcribe(self, wav: np.ndarray, lang: str) -> str:
        import torch
        from conformer_ctc.data import FeatureConfig, audio_to_log_mel

        feats = audio_to_log_mel(torch.from_numpy(wav), FeatureConfig(sample_rate=SR, n_mels=self.cfg.n_mels))
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        lengths = torch.tensor([feats.shape[1]])
        with torch.inference_mode():
            logits, out_lens = self.model(feats.to(self.device), lengths.to(self.device))
        ids = logits[0, : int(out_lens[0])].argmax(-1).tolist()
        blank = getattr(self.cfg, "blank_id", 0)
        pad = getattr(self.cfg, "pad_id", None)
        out, prev = [], None
        for i in ids:
            if i != prev and i != blank and i != pad:
                out.append(self.id2tok.get(i, ""))
            prev = i
        text = "".join(out).replace("|", " ").replace("▁", " ")
        return re.sub(r"\s+", " ", text).strip()


class MMS:
    name = "mms-1b-all"

    def __init__(self, lora_dir: str | None, device: str):
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.device = device
        self.lora_dir = lora_dir
        self.processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device).eval()
        self.lang = None
        if lora_dir:
            self.name = f"mms-1b+lora({Path(lora_dir).name})"

    def _set_lang(self, lang: str):
        if lang != self.lang:
            self.processor.tokenizer.set_target_lang(lang)
            self.model.load_adapter(lang)
            if self.lora_dir:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.lora_dir).eval()
            self.lang = lang

    def transcribe(self, wav: np.ndarray, lang: str) -> str:
        import torch

        self._set_lang(lang)
        inputs = self.processor(wav, sampling_rate=SR, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        return self.processor.decode(logits[0].argmax(-1))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", default=["aka", "ewe", "dag", "gaa"], choices=list(TEST_SETS))
    ap.add_argument("--n", type=int, default=200, help="clips per language")
    ap.add_argument("--models", nargs="+", default=["griot", "mms"], choices=["griot", "mms"])
    ap.add_argument("--griot-dir", default=None, help="local griot-nano-1 snapshot (default: download)")
    ap.add_argument("--mms-lora", default=None, help="fine-tuned LoRA adapter dir for MMS (optional)")
    ap.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    ap.add_argument("--out", default="reports/asr_comparison")
    args = ap.parse_args()

    import jiwer
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    clips = []
    for lang in args.langs:
        t = time.time()
        got = load_test_clips(lang, args.n)
        print(f"[data] {lang}: {len(got)} clips, {sum(c['dur'] for c in got)/60:.1f} min ({time.time()-t:.0f}s)")
        clips += got

    rows = []
    for key in args.models:
        model = Griot(args.griot_dir, args.device) if key == "griot" else MMS(args.mms_lora, args.device)
        print(f"[model] {model.name} on {args.device}")
        for i, c in enumerate(clips):
            t = time.time()
            try:
                hyp = model.transcribe(c["wav"], c["lang"])
            except Exception as e:  # keep going; record the failure
                hyp, err = "", repr(e)
            else:
                err = ""
            rows.append({"model": model.name, "lang": c["lang"], "source": c["source"], "dur": c["dur"],
                         "secs": time.time() - t, "ref": c["ref"], "hyp": hyp, "error": err})
            if i % 50 == 0:
                print(f"  {i}/{len(clips)} {c['lang']} | {hyp[:70]}")
        del model

    df = pd.DataFrame(rows)
    df["ref_n"] = df.ref.map(normalize)
    df["hyp_n"] = df.hyp.map(normalize)
    df.to_csv(out.with_suffix(".csv"), index=False)

    summary = []
    for (m, lang), g in df.groupby(["model", "lang"]):
        g = g[g.ref_n != ""]
        summary.append({
            "model": m, "lang": lang, "clips": len(g),
            "WER": 100 * jiwer.wer(list(g.ref_n), list(g.hyp_n)),
            "CER": 100 * jiwer.cer(list(g.ref_n), list(g.hyp_n)),
            "sec_per_clip": g.secs.mean(), "rtf": g.secs.sum() / g.dur.sum(),
            "errors": int((g.error != "").sum()),
        })
    s = pd.DataFrame(summary).sort_values(["lang", "model"])
    print(s.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    s.to_csv(out.with_name(out.name + "_summary.csv"), index=False)
    return s


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
