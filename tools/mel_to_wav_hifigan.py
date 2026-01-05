import argparse
from pathlib import Path
import json

import numpy as np
import torch
from scipy.io.wavfile import write

from third_party.hifigan.env import AttrDict
from third_party.hifigan.models import Generator


def load_generator(checkpoint_path: str, config_path: str, device: torch.device):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    h = AttrDict(cfg)
    generator = Generator(h).to(device)
    state = torch.load(checkpoint_path, map_location=device)

    # Common checkpoint formats
    if "generator" in state:
        generator.load_state_dict(state["generator"])
    elif "state_dict" in state:
        generator.load_state_dict(state["state_dict"], strict=False)
    else:
        generator.load_state_dict(state, strict=False)

    generator.eval()
    generator.remove_weight_norm()
    return generator, h


def convert_one_mel(gen, h, mel_path: Path, out_path: Path, device: torch.device):
    m = np.load(mel_path)  # (T, n_mels)

    if m.ndim != 2 or m.shape[1] != h.num_mels:
        raise ValueError(
            f"{mel_path}: expected shape (T, {h.num_mels}), got {m.shape}"
        )

    # QWGAN expects (B, n_mels, T)
    mel = torch.from_numpy(m.T).unsqueeze(0).to(device).float()

    with torch.no_grad():
        y = gen(mel)
        y = y.squeeze().cpu().numpy()

    # Normalize to int16 WAV
    y = y / max(1e-8, np.max(np.abs(y)))
    wav16 = (y * 32767.0).astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), int(h.sampling_rate), wav16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mel",
        required=True,
        help="Path to mel .npy OR directory containing *.npy files",
    )
    ap.add_argument("--hifigan", required=True, help="Path to QWGAN generator .pth")
    ap.add_argument("--config", required=True, help="Path to QWGAN config.json")
    ap.add_argument(
        "--out",
        required=True,
        help="Output wav file OR output directory if --mel is a directory",
    )
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    gen, h = load_generator(args.hifigan, args.config, device)

    mel_path = Path(args.mel)
    out_path = Path(args.out)

    # ---- SINGLE MEL ----
    if mel_path.is_file():
        if out_path.is_dir():
            out_path = out_path / (mel_path.stem + ".wav")

        print(f"[QWGAN] Converting {mel_path.name} → {out_path.name}")
        convert_one_mel(gen, h, mel_path, out_path, device)

    # ---- DIRECTORY OF MELS ----
    elif mel_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        mel_files = sorted(mel_path.glob("*.npy"))

        if not mel_files:
            raise RuntimeError(f"No .npy files found in {mel_path}")

        print(f"[QWGAN] Converting {len(mel_files)} mel files...")
        for m in mel_files:
            wav_out = out_path / (m.stem + ".wav")
            print(f"  {m.name} → {wav_out.name}")
            convert_one_mel(gen, h, m, wav_out, device)

    else:
        raise FileNotFoundError(f"{mel_path} does not exist")


if __name__ == "__main__":
    main()
