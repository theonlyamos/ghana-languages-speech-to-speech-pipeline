# Training on Google Colab (CLI)

This guide shows how to train the Ghana S2S models on remote Google Colab GPUs
using the **official [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)**
(`google-colab-cli`, released June 2026). The CLI provisions a GPU/TPU runtime,
runs your code on it, and pulls artifacts back to your machine — all from the
terminal, no browser tab required.

> The pipeline's training surface is the notebook **`ghana_s2s_pipeline.ipynb`**
> (Part 3: Model Training). The CLI runs that notebook — or any script — on a
> Colab GPU instead of your local machine.

---

## ⚠️ Read first

- **Windows is not supported by the Colab CLI.** It runs on **Linux and macOS only**.
  On Windows 11, run everything in this guide inside **WSL2** (Ubuntu) — see
  [Windows: use WSL2](#windows-use-wsl2) below. The CLI runs *locally* and talks to
  a *remote* Colab VM, so WSL2 is all you need; you do not need a local GPU.
- **GPU access is tier-gated.** Most free accounts can only get a **CPU** runtime.
  T4/L4/A100 GPUs generally require **Colab Pro / Pro+ / Pay-As-You-Go**. An
  unstopped session burns compute units until a **24-hour keep-alive cap** — always
  `colab stop` when you finish.
- **The CLI itself needs Python ≥ 3.12** to install. This is separate from the
  project runtime (Python 3.10/3.11), which lives on the *Colab VM*, not your machine.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| OS | Linux or macOS (Windows → WSL2) |
| Python ≥ 3.12 | To install the CLI tool locally |
| Google Cloud SDK (`gcloud`) | For the recommended ADC auth |
| Colab account with GPU access | Pro / Pro+ / Pay-As-You-Go for T4/L4/A100 |
| This repo, cloned locally | Provides `requirements.txt`, the notebook, and `scripts/colab_train.sh` |

### Windows: use WSL2

```powershell
# In an elevated PowerShell (one-time)
wsl --install -d Ubuntu
```

Then open the **Ubuntu** terminal and run all the commands below there. Clone the
repo *inside* WSL (e.g. under `~/ghana_sts_model`) rather than under `/mnt/c/...`
for much faster file I/O.

---

## 2. Install the CLI

```bash
# Recommended: uv (isolated tool install)
uv tool install google-colab-cli

# Or pip
pip install google-colab-cli

# Verify
colab version
```

---

## 3. Authenticate (one time)

The CLI supports two auth strategies. **ADC is recommended** (more reliable,
works headless). Global flags like `--auth` must come **before** the subcommand.

### Option A — Application Default Credentials (recommended)

Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install), then:

```bash
gcloud auth application-default login \
  --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

All four scopes are required: `userinfo.email` (session backend),
`colaboratory` (runtime keep-alive), and `openid`+`cloud-platform` (mandated by
gcloud). ADC is the default, so subsequent commands need no `--auth` flag.

### Option B — OAuth2 (browser consent)

```bash
colab --auth=oauth2 sessions   # triggers a browser consent flow on first use
```

The token is cached at `~/.config/colab-cli/token.json`; the client config is read
from `~/.colab-cli-oauth-config.json`.

### Verify

```bash
colab sessions   # lists active sessions; succeeds once auth works
```

> `colab auth` and `colab drivemount` require a human at a real terminal (TTY) —
> do not script them non-interactively or they will hang.

---

## 4. Choose a GPU for this project

`config.py` is tuned for a **24 GB** card (RTX 3090/4090). Map that to Colab:

| `--gpu` | VRAM | Good for | Notes |
|---------|------|----------|-------|
| `T4`    | 16 GB | Sample-mode smoke tests, NLLB translation setup | Cheapest; reduce `batch_size` and keep `sample_mode=True` |
| `L4`    | 24 GB | **Closest match to the project's target** — ASR + TTS at the default config | Best price/VRAM for full-config runs |
| `A100`  | 40 GB | Fastest ASR/TTS training, larger batches, NLLB-3.3B | Burns compute units fast — always `colab stop` |
| `H100` / `G4` | — | Available if your tier allows | Overkill for this workload |

> An **unrecognized `--gpu` value silently falls back to A100** (and then usually
> fails to allocate). Use one of the exact names above.

**Recommendation:** validate end-to-end on **T4 with `sample_mode=True`** first,
then run full training on **L4** (or **A100** for speed).

---

## 5. Train — the one-command path

A helper script wires the whole flow together: provision → clone repo on the VM →
install deps → run the notebook → download `models/` and a run log → stop.

```bash
# From the repo root (inside WSL on Windows)
bash scripts/colab_train.sh --gpu L4
```

Useful flags (see the script header for all of them):

```bash
bash scripts/colab_train.sh --gpu A100 --session ghana --drive   # mount Drive for persistence
bash scripts/colab_train.sh --gpu T4 --no-stop                   # keep the VM up to inspect it
```

The script is just a thin wrapper over the manual steps in §6 — read it to see
exactly what runs.

---

## 6. Train — the manual steps (what the script does)

```bash
# 6.1 Provision a GPU session named "ghana"
colab new -s ghana --gpu L4
colab status -s ghana

# 6.2 Put this repo on the VM and make it the working directory.
#     Kernel state (incl. CWD + sys.path) PERSISTS across exec calls in a session,
#     so this chdir sticks for every later exec.
echo 'import os, sys, subprocess
subprocess.run(["git","clone","--depth","1",
  "https://github.com/theonlyamos/ghana-languages-speech-to-speech-pipeline",
  "/content/ghana"], check=True)
os.chdir("/content/ghana"); sys.path.insert(0, "/content/ghana")
print("cwd:", os.getcwd())' | colab exec -s ghana

# 6.3 Install Python deps from the LOCAL requirements.txt (CLI reads it locally,
#     installs on the VM via uv). torch + CUDA are already on the Colab image.
colab install -s ghana -r requirements.txt

# 6.4 Install espeak-ng (system package, needed by Coqui TTS) via apt on the VM
echo 'import subprocess
subprocess.run(["apt-get","-qq","install","-y","espeak-ng","libespeak-ng-dev"], check=True)' \
  | colab exec -s ghana

# 6.5 (Optional) Mount Google Drive for dataset / checkpoint persistence.
#     Requires a human at the terminal — run it yourself, not from a script.
colab drivemount -s ghana          # mounts at /content/drive

# 6.6 Run training. `-f` reads the notebook from your LOCAL machine (run this from
#     the repo root) and executes it in this session's kernel — which is still
#     chdir'd into /content/ghana from step 6.2, so the notebook's `Path.cwd()`-based
#     imports (config, utils) resolve. Executing it runs Parts 0–7; outputs are
#     written to ghana_s2s_pipeline_output.ipynb. See §7 re: dataset size first.
colab exec -s ghana -f ghana_s2s_pipeline.ipynb

# 6.7 Retrieve trained models and a replayable log
colab download -s ghana /content/ghana/models ./models
colab log -s ghana -o training_log.ipynb

# 6.8 ALWAYS release the VM
colab stop -s ghana
```

> Kernel state persists across `colab exec` calls in the same session, so imports,
> variables, the working directory, and downloaded data carry over between steps.

---

## 7. Datasets are large — plan storage

The full corpora do **not** fit on a default Colab VM disk:

| Dataset | Size | Strategy on Colab |
|---------|------|-------------------|
| UGSpeechData | ~336 GB | Never download in full to the VM. Mount Drive and stage one language at a time, or download a single `*.tar` per run. |
| BibleTTS | ~50 GB | Stage to Drive; pull only the languages you train. |
| FISD | ~1.2 GB | Fits on the VM directly. |

Practical recipes:

- **First run / smoke test:** keep `config.dataset.sample_mode = True` (the default)
  and a small `sample_size`. This validates ASR → TTS → translation end-to-end
  cheaply on a T4/L4.
- **Full training:** `colab drivemount`, pre-stage the dataset archives to your
  Drive once, and point the data dir at Drive (e.g. symlink
  `data/raw` → `/content/drive/MyDrive/ghana_data`) so downloads survive VM resets.
- **Per-language:** edit `config.dataset.languages` to one language to bound disk
  and time per run.

Because running the whole notebook triggers the download cells, prefer the
sample-mode validation run first, then a Drive-backed full run.

---

## 8. One-shot jobs (`colab run`)

For a single self-contained script (no persistent session), `colab run` provisions
a fresh VM, runs the script with forwarded args, retrieves outputs, and tears the
VM down automatically:

```bash
colab run --gpu T4 scripts/your_train_step.py --epochs 1
```

(`colab run` always provisions its own fresh VM and tears it down afterward, so it
takes no `-s` session flag — the script path and its args follow the options.)

You can even make a script self-executing on a Colab GPU via a shebang:

```bash
#!/usr/bin/env -S colab run --gpu L4 --keep
```

---

## 9. Command cheat-sheet

| Command | Purpose |
|---------|---------|
| `colab new -s NAME [--gpu T4\|L4\|A100\|H100\|G4] [--tpu v5e1\|v6e1]` | Allocate a runtime |
| `colab sessions` | List active sessions |
| `colab status -s NAME` | Hardware / status info |
| `colab exec -s NAME [-f FILE]` | Run stdin / `.py` / `.ipynb` on the VM |
| `colab run [--gpu G] [--keep] SCRIPT [ARGS...]` | Fresh VM → run → auto-teardown |
| `colab repl -s NAME` / `colab console -s NAME` | Interactive Python / raw shell (TTY) |
| `colab install -s NAME [-r FILE \| PKG...]` | Install packages (uv, falls back to pip) |
| `colab upload -s NAME LOCAL REMOTE` | Upload a file to the VM |
| `colab download -s NAME REMOTE LOCAL` | Download a file/dir from the VM |
| `colab ls -s NAME [PATH]` | List remote files |
| `colab drivemount -s NAME [PATH]` | Mount Google Drive (default `/content/drive`, TTY) |
| `colab log -s NAME [-n N] [-o FILE]` | Export history (`.ipynb`/`.md`/`.txt`/`.jsonl`) |
| `colab restart-kernel -s NAME` | Restart the Jupyter kernel |
| `colab stop -s NAME` | Terminate the VM |
| `colab pay` | Open subscription / compute-unit management |
| `colab update [--install]` | Check for / install a new CLI release |

> When only **one** session is active, the `-s NAME` flag is optional.

---

## 10. Troubleshooting

- **Only get a CPU runtime / GPU won't allocate** — accelerator access is tier-gated.
  Upgrade via `colab pay` (Pro/Pro+/Pay-As-You-Go), and double-check the `--gpu`
  spelling (a typo silently falls back to A100 and then fails).
- **A command hangs forever** — `colab auth` / `colab drivemount` need a real
  terminal; never run them from a non-interactive script.
- **`ModuleNotFoundError: utils` (or `config`)** when running the notebook — the
  working directory isn't the repo root. Re-run the §6.2 bootstrap (it `chdir`s into
  the clone) in the *same* session before `colab exec -f`.
- **Disk fills up during dataset download** — see §7; use sample mode or Drive.
- **VM disappeared / out of compute units** — sessions die at the 24h keep-alive cap
  or when units run out. Re-provision and remount Drive. Always `colab stop` to avoid
  silently burning units.

---

## Sources

- [Google Colab CLI — GitHub](https://github.com/googlecolab/google-colab-cli)
- [Colab CLI agent skill / operator reference (`COLAB_SKILL.md`)](https://github.com/googlecolab/google-colab-cli/blob/main/COLAB_SKILL.md)
- [Introducing the Google Colab CLI — Google Developers Blog](https://developers.googleblog.com/introducing-the-google-colab-cli/)
- [`google-colab-cli` on PyPI](https://pypi.org/project/google-colab-cli/)
