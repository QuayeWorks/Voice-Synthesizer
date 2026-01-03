# debug_dataset_mel_stats.py
import torch
import numpy as np

# --- Add project root to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from third_party.tacotron2.hparams import create_hparams
from third_party.tacotron2.data_utils import TextMelLoader

def main():
    hp = create_hparams("")
    print("sampling_rate:", hp.sampling_rate)
    print("mel_fmax:", hp.mel_fmax)
    print("training_files:", hp.training_files)

    loader = TextMelLoader(hp.training_files, hp)

    for idx in range(3):
        # Your loader returns (text, mel)
        text, mel = loader[idx]

        # mel: [n_mels, T]
        mel_np = mel.numpy()

        print(f"\n--- Sample {idx} ---")
        print("mel shape:", mel_np.shape)
        print("mel min:", float(mel_np.min()))
        print("mel max:", float(mel_np.max()))
        print("mel mean:", float(mel_np.mean()))

        flat = mel_np.flatten()
        print("first 20 values:", flat[:20])

if __name__ == "__main__":
    main()
