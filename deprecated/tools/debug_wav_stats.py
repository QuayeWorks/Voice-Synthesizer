# debug_wav_stats.py
import os

# --- Add project root to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from third_party.tacotron2.hparams import create_hparams
from third_party.tacotron2.data_utils import TextMelLoader
from third_party.tacotron2.utils import load_wav_to_torch

def main():
    hp = create_hparams("")
    print("training_files:", hp.training_files)

    loader = TextMelLoader(hp.training_files, hp)

    # Get the path of the first sample directly from filelist
    with open(hp.training_files, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    wav_path, _ = first_line.split("|", 1)
    wav_path = wav_path.strip()
    print("wav_path:", wav_path)

    audio, sr = load_wav_to_torch(wav_path)
    audio = audio.float()
    print("loaded sr:", sr)
    print("audio shape:", audio.shape)
    print("audio min:", float(audio.min()))
    print("audio max:", float(audio.max()))
    print("audio mean:", float(audio.mean()))
    print("audio abs mean:", float(audio.abs().mean()))

if __name__ == "__main__":
    main()

