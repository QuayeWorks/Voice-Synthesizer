# debug_stft_vs_mel.py
import numpy as np
import torch

# --- Add project root to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from third_party.tacotron2.hparams import create_hparams
from third_party.tacotron2.utils import load_wav_to_torch
from third_party.tacotron2.layers import TacotronSTFT
from third_party.tacotron2.stft import STFT


def main():
    hp = create_hparams("")
    print("sampling_rate:", hp.sampling_rate)
    print("mel_fmax:", hp.mel_fmax)
    print("training_files:", hp.training_files)

    # First wav in filelist
    with open(hp.training_files, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    wav_path, _ = first_line.split("|", 1)
    wav_path = wav_path.strip()
    print("wav_path:", wav_path)

    # WAV STATS
    audio, sr = load_wav_to_torch(wav_path)
    audio = audio.float()
    print("\n[WAV]")
    print("loaded sr:", sr)
    print("audio shape:", audio.shape)
    print("audio min / max / mean / abs_mean:",
          float(audio.min()), float(audio.max()),
          float(audio.mean()), float(audio.abs().mean()))

    audio_batch = audio.unsqueeze(0)  # [1, T]

    # RAW STFT MAGNITUDES
    stft = STFT(hp.filter_length, hp.hop_length, hp.win_length)
    mags, phases = stft.transform(audio_batch)
    mags_np = mags.squeeze(0).numpy()  # [n_freq, frames]

    print("\n[STFT magnitudes]")
    print("shape:", mags_np.shape)
    print("min:", float(mags_np.min()))
    print("max:", float(mags_np.max()))
    print("mean:", float(mags_np.mean()))

    # MEL BASIS
    t_stft = TacotronSTFT(
        hp.filter_length, hp.hop_length, hp.win_length,
        hp.n_mel_channels, hp.sampling_rate,
        hp.mel_fmin, hp.mel_fmax
    )
    mel_basis = t_stft.mel_basis.numpy()  # [n_mels, n_freq]

    # Linear mel (before log)
    mel_linear = np.matmul(mel_basis, mags_np)

    print("\n[Mel linear BEFORE log]")
    print("shape:", mel_linear.shape)
    print("min:", float(mel_linear.min()))
    print("max:", float(mel_linear.max()))
    print("mean:", float(mel_linear.mean()))

    # Log-mel (Tacotron compression)
    mel_log = np.log(np.clip(mel_linear, 1e-5, None))

    print("\n[Mel log AFTER compression]")
    print("min:", float(mel_log.min()))
    print("max:", float(mel_log.max()))
    print("mean:", float(mel_log.mean()))
    print("first 20:", mel_log.flatten()[:20])


if __name__ == "__main__":
    main()
