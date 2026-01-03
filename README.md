# Voice Synthesizer (FastPitch + HiFi-GAN)

This repository is trimmed to a FastPitch (text→mel) + HiFi-GAN (mel→wav) pipeline with a lightweight PyQt5 GUI (`run.py`).

## What's inside
- **FastPitch** training and inference (`third_party/fastpitch`).
- **HiFi-GAN** vocoder and configs (`third_party/hifigan`).
- Minimal helper scripts in `tools/` and assets under `inference/`.
- A simplified GUI (`run.py`) that shells out to the FastPitch/HiFi-GAN CLIs.
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

### 2) Train FastPitch
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

### 3) FastPitch inference (text → mel)
```bash
python third_party/fastpitch/inference.py \
  -i inference/phrases/my_lines.txt \
  -o inference/mels \
  --save-mels \
  --fastpitch checkpoints/fastpitch/FastPitch_checkpoint_100.pt \
  --cuda \
  --batch-size 1
```
This writes `.npy` mel files under `inference/mels`. Add `--pace <float>` to adjust speaking speed.

### 4) HiFi-GAN vocoding (mel → wav)
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
Replace checkpoint/config paths with your trained HiFi-GAN assets.

### 5) GUI

Install the GUI/runtime dependencies (PyTorch, PyQt5, NumPy, SciPy):
```bash
pip install -r requirements.txt
```

Launch the GUI:
```bash
python run.py
```

In the window, choose your FastPitch checkpoint, HiFi-GAN generator + config, output directory, and paste one phrase per line. Optional controls include pace, pitch transforms, CUDA/AMP toggles, mel saving, and batch size. Outputs are written to `<output>/mels/*.npy` (when enabled) and `<output>/wavs/*.wav` with unique filenames.

## KEEP vs REMOVED (quarantined)
- **Kept:** FastPitch + HiFi-GAN code, minimal helper scripts (`tools/`), GUI (`run.py`), inference assets, checkpoints folder structure.
- **Quarantined:** Tacotron2/WaveGlow inference script and legacy config (`deprecated/`), unused debug/inspection utilities (`deprecated/tools`), old logs and placeholder inference files, Python `__pycache__` outputs.

## Notes
- Run commands from the repository root so relative imports resolve correctly.
- `inference/mels` and `inference/wavs` stay organized for saved assets; `checkpoints/` holds your trained models.
- `third_party/fastpitch` includes CMUDict assets for English text normalization.
