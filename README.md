# Voice Synthesizer (QWPitch + QWGAN)

This repository is trimmed to a QWPitch (text→mel) + QWGAN (mel→wav) pipeline with a lightweight PyQt5 GUI (`run.py`).

## What's inside
- **QWPitch** training and inference (`third_party/fastpitch`).
- **QWGAN** vocoder and configs (`third_party/hifigan`).
- Minimal helper scripts in `tools/` and assets under `inference/`.
- A simplified GUI (`run.py`) that shells out to the QWPitch/QWGAN CLIs.
- Legacy utilities quarantined under `deprecated/`.

## Quickstart
Assumes the LJSpeech dataset is extracted to `datasets/LJSpeech-1.1`.

## WSL (Ubuntu) setup
Install the base Linux dependencies inside WSL before creating a virtual environment:
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev \
  build-essential libgl1 ffmpeg

python3 -m venv .venv
cd ~/projects/VoiceSynthesizer
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
These packages cover PyTorch builds, PyQt5 GUI support (via `libgl1`), audio processing (`ffmpeg`), and common build tools.

### 1) Prepare dataset
```bash
python tools/build_fastpitch_ljs_filelists.py
python third_party/fastpitch/prepare_dataset.py \
  -d datasets/LJSpeech-1.1 \
  --wav-text-filelists \
    third_party/fastpitch/filelists/ljs_audio_text_train_filelist.txt \
    third_party/fastpitch/filelists/ljs_audio_text_val_filelist.txt \
  --extract-mels --extract-pitch
```

### 2) Train QWPitch
```bash
# Single GPU (simplest)
python third_party/fastpitch/train.py \
  --cuda --epochs 10 -bs 16 -lr 1.5e-4 \
  -d datasets/LJSpeech-1.1 \
  --training-files filelists/ljs_audio_text_train_filelist.txt \
  --validation-files filelists/ljs_audio_text_val_filelist.txt \
  -o checkpoints/fastpitch

# 2 GPUs (DDP)
torchrun --nproc_per_node=2 --master_port=29501 third_party/fastpitch/train.py \
  --cuda --epochs 10 -bs 16 -lr 1.5e-4 \
  -d datasets/LJSpeech-1.1 \
  --training-files filelists/ljs_audio_text_train_filelist.txt \
  --validation-files filelists/ljs_audio_text_val_filelist.txt \
  -o checkpoints/fastpitch
```
Adjust epochs/batch size as hardware allows. Use `--resume` (or `--checkpoint-path`) to continue training from the latest checkpoint.

### 3) QWPitch inference (text → mel)
```bash
python third_party/fastpitch/inference.py \
  -i inference/phrases/my_lines.txt \
  -o inference/mels \
  --save-mels \
  --fastpitch checkpoints/fastpitch/QWPitch_checkpoint_100.pt \
  --cuda \
  --batch-size 1
```
This writes `.npy` mel files under `inference/mels`. Add `--pace <float>` to adjust speaking speed.

### 4) QWGAN vocoding (mel → wav)
```bash
# Single mel -> wav
python tools/mel_to_wav_hifigan.py \
  --mel inference/mels/mel_00000.npy \
  --hifigan checkpoints/hifigan/generator.pth \
  --config third_party/hifigan/config_v1.json \
  --out inference/wavs/mel_00000.wav \
  --cuda

# Folder of mels -> folder of wavs (recommended)
python tools/mel_to_wav_hifigan.py \
  --mel inference/mels \
  --hifigan checkpoints/hifigan/generator.pth \
  --config third_party/hifigan/config_v1.json \
  --out inference/wavs \
  --cuda
```
Replace checkpoint/config paths with your trained QWGAN assets.

### 5) GUI

Install the GUI/runtime dependencies (PyTorch, PyQt5, NumPy, SciPy):
```bash
pip install -r requirements.txt
```

Launch the GUI:
```bash
python run.py
```

In the window, choose your QWPitch checkpoint, QWGAN generator + config, output directory, and paste one phrase per line. Optional controls include pace, pitch transforms, CUDA/AMP toggles, mel saving, and batch size. Outputs are written to `<output>/mels/*.npy` (when enabled) and `<output>/wavs/*.wav` with unique filenames.

> **Note:** If the CMU Pronouncing Dictionary file (`third_party/fastpitch/cmudict/cmudict-0.7b`) is missing, the GUI will download it automatically the first time phoneme conversion is needed. Offline environments can manually place the file at that path from https://github.com/cmusphinx/cmudict.

## KEEP vs REMOVED (quarantined)
- **Kept:** QWPitch + QWGAN code, minimal helper scripts (`tools/`), GUI (`run.py`), inference assets, checkpoints folder structure.
- **Quarantined:** Tacotron2/WaveGlow inference script and legacy config (`deprecated/`), unused debug/inspection utilities (`deprecated/tools`), old logs and placeholder inference files, Python `__pycache__` outputs.

## Notes
- Run commands from the repository root so relative imports resolve correctly.
- `inference/mels` and `inference/wavs` stay organized for saved assets; `checkpoints/` holds your trained models.
- `third_party/fastpitch` includes CMUDict assets for English text normalization.
