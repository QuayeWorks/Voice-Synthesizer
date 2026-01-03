# debug_compare_mel.py
import numpy as np
import librosa
import torch
# --- Add project root to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from third_party.tacotron2.hparams import create_hparams
from third_party.tacotron2.utils import load_wav_to_torch
from third_party.tacotron2.audio_processing import mel_spectrogram

def main():
    hp = create_hparams("")
    with open(hp.training_files, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    wav_path, _ = first_line.split("|", 1)
    wav_path = wav_path.strip()

    # A) Tacotron-style mel (your current pipeline)
    audio_torch, sr_torch = load_wav_to_torch(wav_path)
    audio_torch = audio_torch.float().unsqueeze(0)  # [1, T]
    mel_taco = mel_spectrogram(
        audio_torch,
        n_fft=hp.filter_length,
        num_mels=hp.n_mel_channels,
        sampling_rate=hp.sampling_rate,
        hop_size=hp.hop_length,
        win_size=hp.win_length,
        fmin=hp.mel_fmin,
        fmax=hp.mel_fmax,
        center=False,
    )
    mel_taco = mel_taco.squeeze(0).cpu().numpy()
    print("Tacotron mel: shape", mel_taco.shape, "min", mel_taco.min(), "max", mel_taco.max(), "mean", mel_taco.mean())

    # B) Plain librosa mel (for sanity)
    y_lib, sr_lib = librosa.load(wav_path, sr=hp.sampling_rate)
    mel_lib = librosa.feature.melspectrogram(
        y=y_lib,
        sr=hp.sampling_rate,
        n_fft=hp.filter_length,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        n_mels=hp.n_mel_channels,
        fmin=hp.mel_fmin,
        fmax=hp.mel_fmax,
        center=False,
        power=1.0,
    )
    mel_lib = np.log(np.clip(mel_lib, 1e-5, None))
    print("Librosa mel:  shape", mel_lib.shape, "min", mel_lib.min(), "max", mel_lib.max(), "mean", mel_lib.mean())

if __name__ == "__main__":
    main()
