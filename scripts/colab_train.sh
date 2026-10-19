#!/usr/bin/env bash
#
# colab_train.sh — train the Ghana S2S models on a remote Google Colab GPU
# using the official Google Colab CLI (https://github.com/googlecolab/google-colab-cli).
#
# It provisions a GPU runtime, clones this repo onto the VM, installs the Python
# and system dependencies, runs the training notebook, downloads the trained
# models + a replayable log, and (by default) stops the VM — including on error,
# via an EXIT trap, so a failed run never leaves a billable VM running.
#
# Requirements:
#   - Linux or macOS (on Windows 11, run this inside WSL2 — the CLI does NOT
#     support Windows natively). See COLAB_CLI.md.
#   - `colab` CLI installed and authenticated:  uv tool install google-colab-cli
#   - GPU-capable Colab tier (Pro / Pro+ / Pay-As-You-Go) for T4/L4/A100.
#
# Usage:
#   bash scripts/colab_train.sh [options]
#
# Options:
#   --gpu NAME          Accelerator: T4 | L4 | A100 | H100 | G4   (default: L4)
#   --session NAME      Session name                              (default: ghana)
#   --repo URL          Git URL cloned onto the VM (https:// or git@)
#                       (default: the project's GitHub repo)
#   --branch NAME       Git branch to clone onto the VM          (default: main)
#   --notebook FILE     Notebook to execute (resolved against the repo root)
#                       (default: ghana_s2s_pipeline.ipynb)
#   --f5 NAME           Instead of the notebook, train an F5-TTS voice called NAME
#                       (scripts/train_tts_f5.py): download BibleTTS, prepare, train,
#                       and pull back models/tts_f5/NAME
#   --f5-lang LANG      BibleTTS language for --f5: asante_twi | akuapem_twi | ewe
#                       (default: asante_twi)
#   --f5-epochs N       Training epochs for --f5                  (default: 20)
#   --f5-limit N        Use only N clips for --f5 (quick smoke run; default: all)
#   --requirements FILE Local requirements file installed on the VM
#                       (default: requirements.txt)
#   --drive             Mount Google Drive on the VM (interactive; for dataset/
#                       checkpoint persistence)
#   --no-stop           Leave the VM running after training (inspect it, then
#                       `colab stop -s <session>` yourself)
#   -h, --help          Show this help and exit
#
# Examples:
#   bash scripts/colab_train.sh --gpu L4
#   bash scripts/colab_train.sh --gpu A100 --session ghana --drive
#   bash scripts/colab_train.sh --gpu T4 --no-stop
#   bash scripts/colab_train.sh --gpu L4 --f5 twi --f5-limit 500 --f5-epochs 2   # smoke run
#   bash scripts/colab_train.sh --gpu A100 --f5 twi                              # full Twi voice
#
set -euo pipefail

# ---- defaults --------------------------------------------------------------
GPU="L4"
SESSION="ghana"
REPO="https://github.com/theonlyamos/ghana-languages-speech-to-speech-pipeline"
NOTEBOOK="ghana_s2s_pipeline.ipynb"
BRANCH="main"
F5_NAME=""
F5_LANG="asante_twi"
F5_EPOCHS=20
F5_LIMIT=0
REQUIREMENTS="requirements.txt"
MOUNT_DRIVE=0
STOP_AFTER=1
REMOTE_DIR="/content/ghana"
PROVISIONED=0

# Resolve repo root from this script's location so paths work from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# Print the leading header comment block (between the shebang and the first
# non-comment line), stripping the leading "# ".
usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "${BASH_SOURCE[0]}"; exit 0; }

# Stop the VM on ANY exit (success or failure) once it has been provisioned, so
# an aborted run never silently burns compute units up to the 24h keep-alive cap.
cleanup() {
  local rc=$?
  if [[ "$PROVISIONED" -eq 1 ]]; then
    if [[ "$STOP_AFTER" -eq 1 ]]; then
      log "Releasing the VM (session '$SESSION')..."
      colab stop -s "$SESSION" || warn "Failed to stop '$SESSION' — run manually:  colab stop -s $SESSION"
    else
      warn "Session '$SESSION' left running (--no-stop). Stop it with:  colab stop -s $SESSION"
    fi
  fi
  exit "$rc"
}

# ---- parse args ------------------------------------------------------------
needval() { [[ $# -ge 2 ]] || die "Option '$1' requires a value (use --help)."; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)          needval "$@"; GPU="$2"; shift 2 ;;
    --session)      needval "$@"; SESSION="$2"; shift 2 ;;
    --repo)         needval "$@"; REPO="$2"; shift 2 ;;
    --notebook)     needval "$@"; NOTEBOOK="$2"; shift 2 ;;
    --branch)       needval "$@"; BRANCH="$2"; shift 2 ;;
    --f5)           needval "$@"; F5_NAME="$2"; shift 2 ;;
    --f5-lang)      needval "$@"; F5_LANG="$2"; shift 2 ;;
    --f5-epochs)    needval "$@"; F5_EPOCHS="$2"; shift 2 ;;
    --f5-limit)     needval "$@"; F5_LIMIT="$2"; shift 2 ;;
    --requirements) needval "$@"; REQUIREMENTS="$2"; shift 2 ;;
    --drive)        MOUNT_DRIVE=1; shift ;;
    --no-stop)      STOP_AFTER=0; shift ;;
    -h|--help)      usage ;;
    *)              die "Unknown option: $1 (use --help)" ;;
  esac
done

# ---- preflight -------------------------------------------------------------
command -v colab >/dev/null 2>&1 || die \
  "The 'colab' CLI is not installed. Run: uv tool install google-colab-cli  (see COLAB_CLI.md)"

case "$GPU" in
  T4|L4|A100|H100|G4) ;;
  *) die "Invalid --gpu '$GPU'. Use one of: T4 L4 A100 H100 G4 (an unknown value silently falls back to A100)." ;;
esac

# Whitelist the repo URL — it is spliced into Python that runs on the VM (§clone).
if [[ ! "$REPO" =~ ^(https://|git@)[A-Za-z0-9._:/@~-]+$ ]]; then
  die "Refusing unsafe --repo (expected an https:// or git@ URL with no quotes/spaces): $REPO"
fi

# These are spliced into Python that runs on the VM, so whitelist them too.
[[ "$BRANCH" =~ ^[A-Za-z0-9._/-]+$ ]] || die "Refusing unsafe --branch: $BRANCH"
[[ -z "$F5_NAME" || "$F5_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || die "Invalid --f5 name (letters, digits, _ or -): $F5_NAME"
case "$F5_LANG" in asante_twi|akuapem_twi|ewe) ;; *) die "Invalid --f5-lang '$F5_LANG'. Use asante_twi, akuapem_twi or ewe." ;; esac
[[ "$F5_EPOCHS" =~ ^[0-9]+$ && "$F5_LIMIT" =~ ^[0-9]+$ ]] || die "--f5-epochs / --f5-limit must be whole numbers."

REQ_PATH="$REPO_ROOT/$REQUIREMENTS"
[[ -f "$REQ_PATH" ]] || die "Requirements file not found: $REQ_PATH"

# `colab exec -f` reads a LOCAL file and runs it on the VM, so resolve & check it here.
NB_PATH="$REPO_ROOT/$NOTEBOOK"
[[ -f "$NB_PATH" ]] || die "Notebook not found: $NB_PATH"

log "Verifying Colab connectivity/auth (run 'gcloud auth application-default login' if this fails)..."
colab sessions >/dev/null 2>&1 || die "Could not reach Colab (auth or connectivity). Authenticate per COLAB_CLI.md, then retry."

log "Config: gpu=$GPU session=$SESSION repo=$REPO branch=$BRANCH notebook=$NOTEBOOK f5=${F5_NAME:-off} drive=$MOUNT_DRIVE stop=$STOP_AFTER"

# ---- 1. provision (arm the teardown trap the moment a VM exists) -----------
log "Provisioning '$GPU' runtime as session '$SESSION'..."
trap cleanup EXIT
colab new -s "$SESSION" --gpu "$GPU"
PROVISIONED=1
colab status -s "$SESSION" || true

# ---- 2. clone repo on the VM and pin CWD (persists across exec calls) ------
# REPO is whitelisted above; REMOTE_DIR is a fixed constant, so this unquoted
# heredoc interpolates only vetted values into the remote Python.
log "Cloning repo onto the VM at $REMOTE_DIR ..."
colab exec -s "$SESSION" <<PY
import os, sys, subprocess
if not os.path.isdir("$REMOTE_DIR/.git"):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "$BRANCH", "$REPO", "$REMOTE_DIR"], check=True)
os.chdir("$REMOTE_DIR")
sys.path.insert(0, "$REMOTE_DIR")
print("cwd:", os.getcwd())
PY

# ---- 3. python dependencies (from the LOCAL requirements file) -------------
log "Installing Python dependencies from $REQUIREMENTS (torch + CUDA already on the Colab image)..."
colab install -s "$SESSION" -r "$REQ_PATH"

# ---- 4. system dependency for Coqui TTS ------------------------------------
log "Installing espeak-ng on the VM..."
colab exec -s "$SESSION" <<'PY'
import subprocess
subprocess.run(["apt-get", "-qq", "update"], check=True)
subprocess.run(["apt-get", "-qq", "install", "-y", "espeak-ng", "libespeak-ng-dev"], check=True)
print("espeak-ng installed")
PY

# ---- 5. optional Drive mount (interactive) ---------------------------------
if [[ "$MOUNT_DRIVE" -eq 1 ]]; then
  warn "Mounting Google Drive requires a human at the terminal — follow the prompt."
  colab drivemount -s "$SESSION"
fi

# ---- 6. run training -------------------------------------------------------
if [[ -n "$F5_NAME" ]]; then
  # All values below were whitelisted above.
  log "Training F5-TTS voice '$F5_NAME' on BibleTTS $F5_LANG (this can take hours)..."
  colab exec -s "$SESSION" <<PY
import subprocess, sys
from utils.data_processing import DatasetDownloader
DatasetDownloader().download_bibletis(["$F5_LANG"])
run = lambda *a: subprocess.run([sys.executable, "scripts/train_tts_f5.py", *a], check=True)
run("prepare", "--input", "data/raw/bibletis/$F5_LANG", "--name", "$F5_NAME", "--limit", "$F5_LIMIT")
run("train", "--name", "$F5_NAME", "--epochs", "$F5_EPOCHS")
PY
else
  warn "Running the FULL notebook executes Parts 0-7, including dataset download cells."
  warn "For a first run keep config.dataset.sample_mode = True (default). See COLAB_CLI.md section 7."
  log "Executing $NOTEBOOK on the VM (this can take a long time)..."
  colab exec -s "$SESSION" -f "$NB_PATH"
fi

# ---- 7. retrieve artifacts -------------------------------------------------
if [[ -n "$F5_NAME" ]]; then
  # Only what inference needs (skips numbered checkpoints and the pretrained copy).
  VOICE_REMOTE="$REMOTE_DIR/models/tts_f5/$F5_NAME"
  VOICE_LOCAL="$REPO_ROOT/models/tts_f5/$F5_NAME"
  log "Downloading the trained voice to models/tts_f5/$F5_NAME ..."
  mkdir -p "$VOICE_LOCAL"
  for f in model_last.pt vocab.txt ref.wav ref.txt; do
    colab download -s "$SESSION" "$VOICE_REMOTE/$f" "$VOICE_LOCAL/$f"       || warn "Could not download $f — retrieve manually: colab download -s $SESSION $VOICE_REMOTE/$f $VOICE_LOCAL/$f"
  done
else
  log "Downloading trained models to ./models ..."
  mkdir -p "$REPO_ROOT/models"
  if colab ls -s "$SESSION" "$REMOTE_DIR/models" >/dev/null 2>&1; then
    colab download -s "$SESSION" "$REMOTE_DIR/models" "$REPO_ROOT/models"       || warn "models/ exists on the VM but the download failed — retrieve manually: colab download -s $SESSION $REMOTE_DIR/models ./models"
  else
    warn "No models/ directory on the VM yet (training may not have produced checkpoints)."
  fi
fi

log "Exporting run log to ./training_log.ipynb ..."
colab log -s "$SESSION" -o "$REPO_ROOT/training_log.ipynb" || warn "Could not export the run log."

# ---- 8. done — the EXIT trap (cleanup) releases the VM ----------------------
log "Training run finished."
