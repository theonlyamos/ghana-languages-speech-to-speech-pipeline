"""Checks for the F5-TTS vocab handling. Run: python -m pytest tests/  (or python tests/test_train_tts_f5.py)"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import train_tts_f5 as T  # noqa: E402


def test_vocab_growth():
    # NFC merges "e" + combining acute into one symbol; only unseen symbols are added.
    assert T.normalize("  Mè  yí ") == "Mè yí"
    vocab = [" ", "a", "ɛ"]
    assert T.missing_symbols(["aɛƐ", "a ŋ"], vocab) == ["ŋ", "Ɛ"]


def test_expand_embeddings_keeps_old_rows():
    import torch
    from safetensors.torch import load_file, save_file

    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "a.safetensors"), Path(d, "b.safetensors")
        old = torch.randn(5, 4)
        save_file({T.EMBED_KEY: old, "other": torch.ones(2)}, str(src))
        T.expand_embeddings(str(src), dst, 3)
        new = load_file(str(dst))
        assert new[T.EMBED_KEY].shape == (8, 4)
        assert torch.equal(new[T.EMBED_KEY][:5], old)
        assert torch.equal(new["other"], torch.ones(2))


if __name__ == "__main__":
    test_vocab_growth()
    test_expand_embeddings_keeps_old_rows()
    print("ok")
