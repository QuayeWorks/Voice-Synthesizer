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
python third_party/fastpitch/train.py \
  -o checkpoints/fastpitch \
  -d datasets/LJSpeech-1.1 \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1.5e-4 \
  --training-files third_party/fastpitch/filelists/ljs_audio_text_train_filelist.txt \
  --validation-files third_party/fastpitch/filelists/ljs_audio_text_val_filelist.txt \
  --cuda
```
Adjust epochs/batch size as hardware allows. Use `--resume` or `--checkpoint-path` to continue training.

### 3) FastPitch inference (text → mel)
```bash
python third_party/fastpitch/inference.py \
  -i inference/phrases/my_lines.txt \
  -o inference/mels \
  --save-mels \
  --fastpitch checkpoints/fastpitch/fastpitch_checkpoint.pt \
  --cuda
```
This writes `.npy` mel files under `inference/mels`.

### 4) HiFi-GAN vocoding (mel → wav)
```bash
python tools/mel_to_wav_hifigan.py \
  --mel inference/mels/mel_00000.npy \
  --hifigan checkpoints/hifigan/generator.pth \
  --config third_party/hifigan/config_v1.json \
  --out inference/wavs/mel_00000.wav \
  --cuda
```
Replace checkpoint/config paths with your trained HiFi-GAN assets.

### 5) GUI
```bash
python run.py
```
Fill in the FastPitch/HiFi-GAN checkpoints and press **Synthesize** to run the same pipeline from a desktop window.

## KEEP vs REMOVED (quarantined)
- **Kept:** FastPitch + HiFi-GAN code, minimal helper scripts (`tools/`), GUI (`run.py`), inference assets, checkpoints folder structure.
- **Quarantined:** Tacotron2/WaveGlow inference script and legacy config (`deprecated/`), unused debug/inspection utilities (`deprecated/tools`), old logs and placeholder inference files, Python `__pycache__` outputs.

## Notes
- Run commands from the repository root so relative imports resolve correctly.
- `inference/mels` and `inference/wavs` stay organized for saved assets; `checkpoints/` holds your trained models.
- `third_party/fastpitch` includes CMUDict assets for English text normalization.
