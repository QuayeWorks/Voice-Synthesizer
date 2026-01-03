import os
import sys
import argparse
import json
import numpy as np
import torch

# ----- base paths -----
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TACO_PKG = "third_party.tacotron2"
WG_DIR = os.path.join(ROOT, "third_party", "waveglow")
HIFIGAN_DIR = os.path.join(ROOT, "third_party", "hifigan")

# Make sure waveglow's modules (glow.py, denoiser.py) are importable
if os.path.isdir(WG_DIR) and WG_DIR not in sys.path:
    sys.path.insert(0, WG_DIR)

# HiFi-GAN directory (so its 'env.py' and 'models.py' can be imported as top-level)
if os.path.isdir(HIFIGAN_DIR) and HIFIGAN_DIR not in sys.path:
    sys.path.insert(0, HIFIGAN_DIR)

# ----- tacotron2 imports (package style) -----
from third_party.tacotron2.model import Tacotron2
from third_party.tacotron2.hparams import create_hparams
from third_party.tacotron2.text import text_to_sequence

# ----- waveglow imports -----
from glow import WaveGlow
from denoiser import Denoiser

# ----- optional HiFi-GAN imports -----
# Expecting:
# third_party/hifigan/
#   |-- models.py   (defines Generator)
#   |-- env.py      (defines AttrDict)
try:
    from third_party.hifigan.models import Generator as HifiGenerator
    from third_party.hifigan.env import AttrDict as HifiAttrDict
    _HAS_HIFIGAN = True
except ImportError as e:
    print("[synth] HiFi-GAN import error:", repr(e))
    _HAS_HIFIGAN = False



def _hp_from_ckpt_config(cfg: dict):
    """
    Take a config dict (like the one in tacotron2_latest.pt or our new adapted file)
    and apply it over the default hparams from the NVIDIA repo.
    """
    hp = create_hparams()
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if hasattr(hp, k):
                setattr(hp, k, v)
    return hp


def load_tacotron(path, device="cuda"):
    ckpt = torch.load(path, map_location=device)

    # CASE A: training-style or our new normalized style
    # { "config": {...}, "state_dict": {...} }
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        hparams = _hp_from_ckpt_config(cfg)
        model = Tacotron2(hparams).to(device)

        sd = ckpt["state_dict"]
        # strip "module." if present
        clean_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            clean_sd[k] = v

        missing, unexpected = model.load_state_dict(clean_sd, strict=False)
        if missing:
            print("[synth] missing keys:", missing)
        if unexpected:
            print("[synth] unexpected keys:", unexpected)

        model.eval()
        return model, hparams

    # CASE B: truly bare state_dict (older file)
    else:
        print("[synth] bare state_dict detected, using default hparams")
        hparams = create_hparams()
        model = Tacotron2(hparams).to(device)

        clean_sd = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                k = k[7:]
            clean_sd[k] = v

        missing, unexpected = model.load_state_dict(clean_sd, strict=False)
        if missing:
            print("[synth] missing keys:", missing)
        if unexpected:
            print("[synth] unexpected keys:", unexpected)

        model.eval()
        return model, hparams


def load_waveglow(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        waveglow = ckpt["model"]
    else:
        state = ckpt.get("state_dict", ckpt)
        waveglow = WaveGlow()
        clean_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[7:]
            # often these buffers don’t need to be loaded
            if k.endswith(".num_batches_tracked"):
                continue
            clean_state[k] = v
        waveglow.load_state_dict(clean_state, strict=False)

    waveglow = waveglow.to(device).eval()
    denoiser = Denoiser(waveglow).to(device)
    return waveglow, denoiser


def load_hifigan(generator_path: str, config_path: str, device: str):
    """
    Load a HiFi-GAN generator from a local checkpoint + JSON config.

    Expected:
      - third_party/hifigan/models.py  defines Generator
      - third_party/hifigan/env.py     defines AttrDict
    """
    if not _HAS_HIFIGAN:
        raise RuntimeError(
            "HiFi-GAN not available: models/env not importable. "
            "Make sure you cloned HiFi-GAN into third_party/hifigan."
        )
    if config_path is None:
        raise ValueError("HiFi-GAN requires --hifigan_config pointing to a JSON config.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    hparams = HifiAttrDict(cfg)

    model = HifiGenerator(hparams).to(device)
    state = torch.load(generator_path, map_location=device)
    # common pattern: checkpoint["generator"]
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    model.load_state_dict(state, strict=False)
    model.eval()
    # remove weight norm if the implementation supports it
    if hasattr(model, "remove_weight_norm"):
        model.remove_weight_norm()

    return model


def text_to_tensor(text: str, cleaners):
    seq = text_to_sequence(text, cleaners)
    if not seq:
        seq = [0]
    return torch.LongTensor(seq)[None, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tacotron", required=True, help="path to tacotron2 .pt/.pth")

    # Vocoder selection
    p.add_argument(
        "--vocoder",
        choices=["waveglow", "hifigan"],
        default="waveglow",
        help="Which vocoder to use: 'waveglow' (default) or 'hifigan'.",
    )
    # WaveGlow-specific
    p.add_argument("--waveglow", help="path to waveglow_256.pt (when using --vocoder waveglow)")
    p.add_argument("--denoise", type=float, default=0.0, help="waveglow denoiser strength")
    p.add_argument("--sigma", type=float, default=0.8, help="waveglow noise scale")

    # HiFi-GAN-specific
    p.add_argument("--hifigan", help="path to HiFi-GAN generator checkpoint (when using --vocoder hifigan)")
    p.add_argument("--hifigan_config", help="path to HiFi-GAN JSON config (when using --vocoder hifigan)")

    # Common
    p.add_argument("--text", required=True)
    p.add_argument("--out", default="out.wav")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device
    if not torch.cuda.is_available() or device != "cuda":
        device = "cpu"
    print(f"[synth] using device: {device}")
    print(f"[synth] vocoder: {args.vocoder}")

    # ----- load tacotron -----
    tacotron, hp = load_tacotron(args.tacotron, device)

    # ----- load vocoder -----
    vocoder_type = args.vocoder
    waveglow = None
    denoiser = None
    hifigan = None

    if vocoder_type == "waveglow":
        if not args.waveglow:
            raise ValueError("You must provide --waveglow when using --vocoder waveglow")
        waveglow, denoiser = load_waveglow(args.waveglow, device)
    elif vocoder_type == "hifigan":
        if not args.hifigan:
            raise ValueError("You must provide --hifigan when using --vocoder hifigan")
        if not args.hifigan_config:
            raise ValueError("You must provide --hifigan_config when using --vocoder hifigan")
        hifigan = load_hifigan(args.hifigan, args.hifigan_config, device)
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    cleaners = getattr(hp, "text_cleaners", ["english_cleaners"])
    seq = text_to_tensor(args.text, cleaners).to(device)

    with torch.no_grad():
        print("[dbg] seq ids:", seq.tolist())
        print("[dbg] seq len:", seq.shape[1])
        out = tacotron.inference(seq)

        # NVIDIA Tacotron2 returns (mel_outputs, mel_outputs_postnet, gate_outputs, alignments)
        if isinstance(out, (list, tuple)):
            mel = out[1]
            align = out[3]
        else:
            mel = out.mel_outputs_postnet
            align = out.alignments

        np.save("debug_mel.npy", mel.cpu().numpy())
        np.save("debug_align.npy", align.cpu().numpy())
        print("[synth] align shape:", align.shape)

        
        # --- vocoder inference ---
        if vocoder_type == "waveglow":
            audio = waveglow.infer(mel, sigma=args.sigma).squeeze(0)  # [T]
            if args.denoise > 0.0:
                audio = denoiser(audio.unsqueeze(0), args.denoise).squeeze(0)
        else:  # HiFi-GAN
            # HiFi-GAN returns [B, 1, T]; squeeze to [T]
            audio = hifigan(mel)
            # make sure it's 1D: [T]
            audio = audio.squeeze()  

        # Clamp to a safe range and move to numpy
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        audio = audio.cpu().numpy().astype("float32")


    sr = int(getattr(hp, "sampling_rate", 22050))
    try:
        import soundfile as sf
        sf.write(args.out, audio, sr, subtype="PCM_16")
    except Exception:
        from scipy.io.wavfile import write
        write(args.out, sr, (audio * 32767.0).astype("int16"))

    print(f"[synth] wrote {args.out}")


if __name__ == "__main__":
    main()
