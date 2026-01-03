#venv\Scripts\activate
# app.py — Full desktop TTS workstation UI
# - Synthesis (Tacotron2 + WaveGlow; Griffin–Lim fallback)
# - Dataset tools (Record mic → auto-segment → Whisper transcript → metadata.csv)
# - Training launcher with progress + options (batch size, workers, LR, cuDNN benchmark)
# - Speaker adaptation (quick single-speaker fine-tune)
# - Transcripts viewer
#
# Notes:
# * Expects a local `tts_model` module providing:
#   - train_model(config_path, output_path, progress_callback)
#   - synthesize(text, out_path, model_path, use_cuda=True)   [optional path for custom model]
#   - update_metadata_file(wav_name, transcript) -> returns LJSpeech-style id (e.g., LJ001-0001)
#   - load_wav(path, sr=22050) -> torch.Tensor
#   - compute_mel_spectrogram(wav_tensor, sr=22050) -> torch.Tensor[n_mels, T]
#   - adapt_speaker(model, adaptation_pairs, char_to_idx, num_epochs, lr) -> model
#   - Tacotron2 class (if you want to instantiate your own)
#
# * Whisper is used for ASR on recorded segments (model="medium" by default).
# * If WaveGlow is not provided, Griffin–Lim is used to reconstruct audio from the mel.
# * Designed to run on Windows as well (guards workers=0).
#
# ------------------------------------------------------------
import subprocess
import os
import sys
import io
import json
import time
import math
import shutil
import queue
import contextlib
from dataclasses import dataclass
from typing import Optional, List, Tuple
# --- dataset tools ---
import re, csv, shutil, math, time
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import logging, logging.handlers, datetime, traceback
# whisper (local)
try:
    import whisper
    HAVE_WHISPER = True
except Exception:
    HAVE_WHISPER = False

# ----------------- Silence noisy warnings (optional but nice) -----------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    import torch
    warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
    warnings.filterwarnings("ignore", message=".*torch.cuda.*Tensor constructors are no longer recommended.*")
except Exception:
    pass


# --- add local NVIDIA repo paths to sys.path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # folder of Test_app.py
ROOT_DIR = BASE_DIR                                        # if this file sits at project root
T2 = os.path.join(ROOT_DIR, "third_party", "tacotron2")
WG = os.path.join(ROOT_DIR, "third_party", "waveglow")
for p in (T2, WG):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------- Third-party imports ----------------------------
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets

import numpy as np

# Audio IO for save/fallback
try:
    import soundfile as sf
except Exception:
    sf = None

# Prefer torchaudio for Griffin–Lim and resampling
try:
    import torchaudio
    TORCHAUDIO_OK = True
except Exception:
    TORCHAUDIO_OK = False

# librosa fallback for resampling and griffin-lim (if torchaudio missing)
try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

# Whisper ASR for recorded segments
try:
    import whisper  # openai/whisper
    WHISPER_OK = True
except Exception:
    WHISPER_OK = False

# Mic recording
try:
    import pyaudio
    PYAUDIO_OK = True
except Exception:
    PYAUDIO_OK = False

# Local TTS helpers (user-provided module)
try:
    import tts_model
except Exception as e:
    print("Warning: tts_model not found or failed to import:", e)
    tts_model = None

# ------------------------------------------------------------
# Utility: paths, config, dataset root
# ------------------------------------------------------------
# --- logging bootstrap (writes logs/ui_YYYYmmdd_HHMMSS.log) ---
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(
    LOG_DIR, f"ui_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logger = logging.getLogger("voicesynth")
logger.setLevel(logging.DEBUG)

_fh = logging.handlers.RotatingFileHandler(
    LOG_PATH, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

APP_SR = 22050
DEFAULT_DATASET_ROOT = os.path.join("datasets", "my_dataset")
DEFAULT_CHECKPOINTS_DIR = "checkpoints"
SYNTH_OUT_WAV = "synthesized_audio.wav"

LJ_RE = re.compile(r"^LJ(\d{3})-(\d{3})\.wav$", re.IGNORECASE)

# --- Config I/O ---
import json
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

### CHANGED: expand DEFAULT_CONFIG["model"] to include NVIDIA-style fields
DEFAULT_CONFIG = {
    "audio": {
        "max-wav-value": 32768.0,
        "sampling-rate": 22050,
        "filter-length": 1024,
        "hop-length": 256,
        "win-length": 1024,
        "mel-fmin": 0.0,
        "mel-fmax": 8000.0,
    },
    "data": {
        "training-files": "filelists/myset_train.txt",
        "validation-files": "filelists/myset_val.txt",
        "text-cleaners": ["english_cleaners"],
        "batch-size": 32,
        "iters-per-checkpoint": 1000,
        "epochs": 1500,
        "shuffle-seed": 1234
    },
    "optimization": {
        "learning-rate": 1e-3,
        "weight-decay": 1e-6,
        "grad-clip-thresh": 1.0,
        "fp16-run": False,
        "cudnn-benchmark": False
    },
    "model": {
        "n-mel-channels": 80,
        "n-symbols": 148,                       # added
        "symbols-embedding-dim": 512,
        "encoder-kernel-size": 5,               # added
        "encoder-n-convolutions": 3,            # added
        "encoder-embedding-dim": 512,           # added
        "attention-rnn-dim": 1024,              # added
        "attention-dim": 128,                   # added
        "attention-location-n-filters": 32,     # added
        "attention-location-kernel-size": 31,   # added
        "n-frames-per-step": 1,                 # added
        "decoder-rnn-dim": 1024,                # added
        "prenet-dim": 256,                      # added
        "max-decoder-steps": 1000,              # added
        "gate-threshold": 0.5,                  # added
        "p-attention-dropout": 0.1,             # added
        "p-decoder-dropout": 0.1,               # added
        "postnet-embedding-dim": 512,           # added
        "postnet-kernel-size": 5,               # added
        "postnet-n-convolutions": 5,            # added
        "decoder-no-early-stopping": False,     # added
        "mask-padding": False,                  # added
        "ignore-layers": ["embedding.weight"]
    },
    "distributed": {
        "n-gpus": 1,
        "distributed-run": False,
        "dist-backend": "nccl",
        "dist-url": "tcp://localhost:54321"
    },
    "adaptation": {
        "enable": False,
        "warm-start": "",        # path to teacher ckpt (tacotron2)
        "freeze-encoder": False,
        "freeze-postnet": False,
        "lr-mult": 1.0,          # scale LR for adaptation
        "transfer-layers": []    # e.g., ["embedding", "encoder"]
    }
}

def _install_global_excepthook():
    def _handle(exc_type, exc, tb):
        logging.getLogger("voicesynth").error(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )
    sys.excepthook = _handle
    try:
        from PyQt5 import QtCore
        def _qt_msg(mode, context, message):
            logging.getLogger("voicesynth.qt").warning(
                f"{message} ({context.file}:{context.line})"
            )
        QtCore.qInstallMessageHandler(_qt_msg)
    except Exception:
        pass
        
_install_global_excepthook() 
       
def load_config():
    cfg = DEFAULT_CONFIG.copy()
    # deep-merge
    def _merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and k in a and isinstance(a[k], dict):
                _merge(a[k], v)
            else:
                a[k] = v
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                disk = json.load(f)
                _merge(cfg, disk)
            except Exception:
                pass
    return cfg

def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


def is_lj_name(name:str):
    m = LJ_RE.match(name)
    if not m: return None
    a,b = int(m.group(1)), int(m.group(2))
    return a,b

def next_lj_name(used:set):
    """Yield next available LJ###-###.wav not in used."""
    # flatten used into numeric tuple set
    s = set(used)
    a,b = 0,1
    while True:
        name = f"LJ{a:03d}-{b:03d}.wav"
        if name not in s:
            s.add(name)
            yield name
        # advance
        b += 1
        if b > 999:
            b = 1
            a += 1
            if a > 999:
                raise RuntimeError("Exhausted LJ000-001 .. LJ999-999")

def normalize_text(t:str):
    t = t.replace("\u00A0"," ").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _read_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: str, obj):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def get_active_dataset_root() -> str:
    cfg = _read_json("config.json", {})
    try:
        ds = cfg.get("datasets", [])
        if ds and isinstance(ds, list) and isinstance(ds[0], dict) and ds[0].get("path"):
            return ds[0]["path"]
    except Exception:
        pass
    return DEFAULT_DATASET_ROOT

def ensure_dataset_dirs(root: str):
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "wavs"), exist_ok=True)
    meta = os.path.join(root, "metadata.csv")
    if not os.path.exists(meta):
        with open(meta, "a", encoding="utf-8") as f:
            pass

def dataset_paths():
    root = get_active_dataset_root()
    ensure_dataset_dirs(root)
    return root, os.path.join(root, "wavs"), os.path.join(root, "metadata.csv")

def autodiscover_checkpoint(patterns=(".pt", ".pth", ".pth.tar")) -> Optional[str]:
    cdir = DEFAULT_CHECKPOINTS_DIR
    if not os.path.isdir(cdir):
        return None
    candidates = []
    for fn in os.listdir(cdir):
        if any(fn.endswith(ext) for ext in patterns):
            candidates.append(os.path.join(cdir, fn))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

# ------------------------------------------------------------
# Robust checkpoint loading for Tacotron2 / WaveGlow
# ------------------------------------------------------------

def safe_load_checkpoint(path, map_location=None):
    """Load a PyTorch checkpoint handling common shapes:
       - nested under 'state_dict'
       - keys with or without 'module.' prefix
       Returns: raw object (could be dict) and a flat state_dict if found.
    """
    import torch
    with contextlib.redirect_stdout(io.StringIO()):
        ckpt = torch.load(path, map_location=map_location)

    state = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            # might already be a state dict
            state = ckpt
    return ckpt, state

def _strip_module_prefix(state_dict):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()}

# ------------------------------------------------------------
# NVIDIA Tacotron2 + WaveGlow wrapper (optional; can use your own)
# ------------------------------------------------------------
# --- NVIDIA text helpers ---
from text import symbols as NV_SYMBOLS
from text import text_to_sequence as nv_text_to_sequence

def _as_list_lit(items):
    """Return a python-literal list string like "['a','b']" from list/str."""
    if isinstance(items, str):
        items = [s.strip() for s in items.split(",") if s.strip()]
    return "[" + ",".join(repr(x) for x in items) + "]" if items else "[]"

def _ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    return [x]


def _text_to_sequence_nv(text: str, cleaners):
    seq = nv_text_to_sequence(text, cleaners)
    if not seq:
        seq = [0]
    return torch.LongTensor(seq)

### CHANGED: helper to build hparams from checkpoint config or from our UI config
def _apply_taco_config_to_hparams(hp, cfg_dict: dict):
    """Given a Tacotron hparams object and a dict (like what you saw in the ckpt),
    copy over matching fields."""
    if not isinstance(cfg_dict, dict):
        return hp
    for k, v in cfg_dict.items():
        # some keys may have dashes in JSON, handle both
        if hasattr(hp, k):
            setattr(hp, k, v)
        else:
            # try dashed-to-underscored
            k2 = k.replace("-", "_")
            if hasattr(hp, k2):
                setattr(hp, k2, v)
    return hp
        
class NvSynth:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", denoise_strength=0.0):
        self.device = torch.device(device)
        self.taco = None
        self.waveglow = None
        self.denoiser = None
        self.denoise_strength = float(denoise_strength)
        self.sr = APP_SR
        self._symbols = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;?&"  # minimal set


    def load_tacotron2(self, path: str):
        import torch
        from model import Tacotron2
        from hparams import create_hparams

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Tacotron2 checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device)

        # ### CHANGED:
        # If checkpoint contains a 'config' like your adapted one, use it to build hparams
        hp = create_hparams()
        if isinstance(ckpt, dict) and "config" in ckpt:
            hp = _apply_taco_config_to_hparams(hp, ckpt["config"])
        else:
            # fallback to default NVIDIA-like
            hp.n_symbols = len(NV_SYMBOLS)

        self.sr = int(getattr(hp, "sampling_rate", 22050))

        # Build model and move to device
        model = Tacotron2(hp).to(self.device)

        # Load checkpoint, handle common layouts
        state = ckpt.get("state_dict", ckpt)

        cleaned = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[7:]
            if k.endswith(".num_batches_tracked"):
                continue
            cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print("[taco] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("[taco] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

        model.eval()
        self.taco = model
        self.hp = hp
        
    def load_waveglow(self, path: str, sigma: float = 0.8):
        import os, torch
        from denoiser import Denoiser

        # try both layouts
        try:
            from third_party.waveglow.glow import WaveGlow
        except Exception:
            from waveglow.glow import WaveGlow

        if not os.path.isfile(path):
            raise FileNotFoundError(f"WaveGlow checkpoint not found: {path}")

        # allowlist class for new torch (won't hurt on older)
        try:
            torch.serialization.add_safe_globals([WaveGlow])
        except Exception:
            pass

        # load EXACTLY the file you picked
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # --- reconstruct model from various checkpoint styles ---
        if isinstance(ckpt, WaveGlow):
            waveglow = ckpt
        elif isinstance(ckpt, dict) and "model" in ckpt:
            # common NVIDIA style
            waveglow = ckpt["model"]
        else:
            # probably a state_dict
            state = ckpt.get("state_dict", ckpt)
            waveglow = WaveGlow()
            cleaned = {}
            for k, v in state.items():
                if k.startswith("module."):
                    k = k[7:]
                if k.endswith(".num_batches_tracked"):
                    continue
                cleaned[k] = v
            waveglow.load_state_dict(cleaned, strict=False)

        # --- local helper: remove weightnorm everywhere if present ---
        def _strip_weightnorm(m):
            import torch.nn.utils as nn_utils
            for name, sub in list(m.named_children()):
                _strip_weightnorm(sub)
            # try to remove on this module too
            for wn_name in ("weight", "weight_g", "weight_v"):
                try:
                    nn_utils.remove_weight_norm(m)
                    break
                except Exception:
                    pass

        _strip_weightnorm(waveglow)

        waveglow.to(self.device).eval()
        self.waveglow = waveglow
        self.waveglow_sigma = float(sigma)

        # make denoiser on same device
        self.denoiser = Denoiser(self.waveglow).to(self.device).eval()


    def _griffin_lim_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Reconstruct waveform from mel using Griffin–Lim."""
        n_fft = 1024
        hop = 256
        win = 1024
        device = mel.device
        if TORCHAUDIO_OK:
            fb = torchaudio.functional.create_fb_matrix(n_freqs=n_fft // 2 + 1,
                                                        n_mels=mel.size(1),
                                                        sample_rate=self.sr,
                                                        f_min=0.0, f_max=None, norm=None)
            fb = fb.to(device)
            # mel: [1, 80, T] -> [80, T]
            M = mel[0]
            # pseudo-inverse mel → linear magnitude
            pinv = torch.linalg.pinv(fb)
            linear = torch.clamp(pinv @ M, min=0.0)
            wav = torchaudio.functional.griffinlim(linear, n_fft=n_fft, hop_length=hop, win_length=win)
            return wav
        elif LIBROSA_OK:
            M = mel[0].detach().cpu().numpy()
            linear = librosa.feature.inverse.mel_to_stft(M, sr=self.sr, n_fft=n_fft, power=1.0)
            wav = librosa.griffinlim(linear, hop_length=hop, win_length=win, n_fft=n_fft)
            return torch.tensor(wav, device=device)
        else:
            raise RuntimeError("Neither torchaudio nor librosa available for Griffin–Lim.")

    def infer(self, text: str, out_path: Optional[str] = None, sigma: float = 0.8) -> torch.Tensor:
        assert self.taco is not None, "Tacotron2 not loaded."
        cleaners = getattr(self.hp, "text_cleaners", ["english_cleaners"])

        # sequence
        seq = _text_to_sequence_nv(text, cleaners).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.taco.inference(seq)
            # handle both tuple and object returns
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                mel = out[1]   # mel_outputs_postnet
            elif hasattr(out, "mel_outputs_postnet"):
                mel = out.mel_outputs_postnet
            else:
                # some forks return (mel, gate, align) – try first element
                mel = out[0] if isinstance(out, (list, tuple)) else out
            # ensure [B, n_mels, T]
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

        # WaveGlow or Griffin-Lim
        if self.waveglow is None:
            wav = self._griffin_lim_from_mel(mel)
        else:
            with torch.no_grad():
                audio = self.waveglow.infer(mel, sigma=sigma)
                # normalize shapes:
                # infer() can return [1, 1, T] or [1, T]; make it [1, T]
                if audio.dim() == 3 and audio.size(1) == 1:
                    audio = audio.squeeze(1)            # [1, T]
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)          # [1, T]
                if self.denoiser is not None and self.denoise_strength > 0:
                    audio = self.denoiser(audio.to(self.device), strength=self.denoise_strength).to("cpu")
                wav = audio.squeeze(0)  

        # Peak-normalize to -3 dBFS
        if torch.is_tensor(wav):
            peak = float(wav.abs().max().item())
        else:
            peak = float(np.max(np.abs(wav)))
        if peak > 0:
            gain = (10 ** (-3/20)) / peak
            wav = wav * gain

        # Optional save (1-D float32)
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            if 'torchaudio' in sys.modules:
                import torchaudio
                torchaudio.save(out_path, wav.detach().cpu().unsqueeze(0), sample_rate=self.sr)
            elif sf is not None:
                sf.write(out_path, wav.detach().cpu().numpy().astype(np.float32), self.sr)

        return wav


# ------------------------------------------------------------
# Worker threads
# ------------------------------------------------------------
class TrainWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, cmd, cwd=None, env=None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.env = env or os.environ.copy()
        self._proc = None
        self._stop = False

    def stop(self):
        self._stop = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def run(self):
        try:
            self.progress.emit("Launching: " + " ".join(self.cmd))
            logging.getLogger("voicesynth.train").info("CMD: %s", " ".join(self.cmd))
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            for line in self._proc.stdout:
                if self._stop:
                    break
                line = line.rstrip()
                self.progress.emit(line)
                logging.getLogger("voicesynth.train").info(line)
            rc = self._proc.wait()
            if self._stop:
                self.failed.emit("Stopped.")
                logging.getLogger("voicesynth.train").warning("Stopped by user.")
            elif rc == 0:
                self.finished.emit("Training finished.")
                logging.getLogger("voicesynth.train").info("Finished successfully.")
            else:
                self.failed.emit(f"Exited with code {rc}.")
                logging.getLogger("voicesynth.train").error("Exited with code %s", rc)
        except Exception as e:
            self.failed.emit(f"Exception: {e}")


class DatasetWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)      # message
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, dataset_dir:str, out_filelists:str, resample_22050:bool, val_ratio:float=0.02, whisper_model:str="small"):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.wavs_dir    = (self.dataset_dir / "wavs")
        self.meta_csv    = (self.dataset_dir / "metadata.csv")
        self.out_filelists = Path(out_filelists)
        self.resample = resample_22050
        self.val_ratio = val_ratio
        self.whisper_model = whisper_model

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if not self.wavs_dir.is_dir():
                raise RuntimeError(f"wavs folder not found: {self.wavs_dir}")
            self.out_filelists.mkdir(parents=True, exist_ok=True)

            # 1) collect wavs
            wavs = sorted([p for p in self.wavs_dir.rglob("*.wav") if p.is_file()])
            if not wavs: raise RuntimeError("No .wav files found.")

            # 2) compute used LJ names
            used = set()
            for p in wavs:
                nm = p.name
                if is_lj_name(nm):
                    used.add(nm)
            name_gen = next_lj_name(used)

            # 3) rename non-compliant files (in place, stable)
            renames = {}
            for p in wavs:
                if not is_lj_name(p.name):
                    new_name = next(name_gen)
                    dst = p.with_name(new_name)
                    # avoid collisions by using temp if needed
                    tmp = p.with_name(f"__tmp__{time.time_ns()}__.wav")
                    p.rename(tmp); tmp.rename(dst)
                    renames[p] = dst
            # refresh list
            wavs = sorted([p for p in self.wavs_dir.iterdir() if p.suffix.lower()==".wav"])

            # 4) optional resample to 22050 (overwrite in place)
            if self.resample:
                for p in wavs:
                    self.progress.emit(f"Resampling: {p.name}")
                    y, sr = sf.read(str(p), always_2d=False)
                    if sr != 22050:
                        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=22050, res_type="kaiser_best")
                        sf.write(str(p), y, 22050, subtype="PCM_16")

            # 5) load whisper
            if not HAVE_WHISPER:
                raise RuntimeError("whisper not installed. pip install openai-whisper and ensure ffmpeg is on PATH.")
            self.progress.emit(f"Loading Whisper model: {self.whisper_model}")
            model = whisper.load_model(self.whisper_model)  # uses CUDA if available

            # 6) transcribe + build rows
            rows = []
            for idx, p in enumerate(sorted(wavs), 1):
                self.progress.emit(f"Transcribing ({idx}/{len(wavs)}): {p.name}")
                # whisper returns dict; we pull ['text']
                result = model.transcribe(str(p), language="en", fp16=torch.cuda.is_available())
                text = normalize_text(result.get("text",""))
                if not text:
                    text = " "
                file_id = p.stem  # e.g., LJ000-001
                rows.append((file_id, text))

            # 7) write metadata.csv (id|raw|norm with same text in both cols)
            with open(self.meta_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="|")
                for fid, t in rows:
                    w.writerow([fid, t, t])

            # 8) write Tacotron2 filelists (abs_path|text)
            lines = []
            for fid, t in rows:
                wav_abs = str((self.wavs_dir / f"{fid}.wav").resolve()).replace("\\","/")
                lines.append(f"{wav_abs}|{t}")

            # split
            n_val = max(1, int(len(lines)*self.val_ratio))
            val_lines = lines[:n_val]
            train_lines = lines[n_val:]

            (self.out_filelists / "myset_train.txt").write_text("\n".join(train_lines)+"\n", encoding="utf-8")
            (self.out_filelists / "myset_val.txt").write_text("\n".join(val_lines)+"\n", encoding="utf-8")

            self.finished.emit(f"Done. {len(lines)} clips. metadata.csv + filelists written.")
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class SynthWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(str)  # path
    error = QtCore.pyqtSignal(str)

    def __init__(self, text, taco_path, wg_path, device, denoise_strength, sigma, out_path):
        super().__init__()
        self.text = text
        self.taco_path = taco_path
        self.wg_path = wg_path
        self.device = device
        self.denoise_strength = float(denoise_strength)
        self.sigma = float(sigma)
        self.out_path = out_path

    def run(self):
        try:
            self.progress.emit(5, "Initializing synthesizer...")
            nv = NvSynth(device=self.device, denoise_strength=self.denoise_strength)
            if not self.taco_path or not os.path.exists(self.taco_path):
                raise RuntimeError("Tacotron2 checkpoint path is invalid.")
            nv.load_tacotron2(self.taco_path)
            self.progress.emit(35, "Tacotron2 loaded.")

            if self.wg_path and os.path.exists(self.wg_path):
                nv.load_waveglow(self.wg_path)
                self.progress.emit(55, "WaveGlow loaded.")
            else:
                self.progress.emit(55, "No WaveGlow provided; using Griffin–Lim.")

            wav = nv.infer(self.text, out_path=self.out_path, sigma=self.sigma)
            self.progress.emit(100, "Synthesis complete.")
            self.finished.emit(self.out_path)
        except Exception as e:
            self.error.emit(f"Synthesis failed: {e}")

class RecorderThread(QtCore.QThread):
    recording_started = QtCore.pyqtSignal()
    recording_stopped = QtCore.pyqtSignal(str)  # path to large recording
    progress = QtCore.pyqtSignal(int, str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, dst_folder, sr=44100):
        super().__init__()
        self.dst_folder = dst_folder
        self.sr = sr
        self._running = False
        self._frames = []
        self._tmp_path = os.path.join(dst_folder, "_recording_raw.wav")

    def run(self):
        if not PYAUDIO_OK:
            self.error.emit("PyAudio not available.")
            return
        pa = pyaudio.PyAudio()
        fmt = pyaudio.paInt16
        ch = 1
        chunk = 1024
        stream = None
        try:
            os.makedirs(self.dst_folder, exist_ok=True)
            self._frames = []
            self._running = True
            stream = pa.open(format=fmt, channels=ch, rate=self.sr, input=True, frames_per_buffer=chunk)
            self.recording_started.emit()
            self.progress.emit(0, "Recording microphone...")

            while self._running:
                data = stream.read(chunk, exception_on_overflow=False)
                self._frames.append(data)
                if len(self._frames) % 50 == 0:
                    self.progress.emit(0, f"Recording... {len(self._frames)*chunk/self.sr:.1f}s")

            # Stop
            stream.stop_stream()
            stream.close()
            # Save the big recording
            import wave
            with wave.open(self._tmp_path, "wb") as wf:
                wf.setnchannels(ch)
                wf.setsampwidth(pa.get_sample_size(fmt))
                wf.setframerate(self.sr)
                wf.writeframes(b"".join(self._frames))

            self.recording_stopped.emit(self._tmp_path)
        except Exception as e:
            self.error.emit(f"Recording error: {e}")
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    def stop(self):
        self._running = False

class TrainingThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, config_obj, out_dir):
        super().__init__()
        self.config_obj = config_obj
        self.out_dir = out_dir

    def run(self):
        if tts_model is None or not hasattr(tts_model, "train_model"):
            self.error.emit("tts_model.train_model not available.")
            return
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            temp_cfg = os.path.join(self.out_dir, "temp_config.json")
            _write_json(temp_cfg, self.config_obj)
            def cb(pct, msg):
                pct = int(max(0, min(100, pct)))
                self.progress.emit(pct, msg)
            tts_model.train_model(temp_cfg, self.out_dir, progress_callback=cb)
            self.finished.emit(self.out_dir)
        except Exception as e:
            self.error.emit(f"Training failed: {e}")

# ------------------------------------------------------------
# Main Window
# ------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tacotron2 Synthesizer — Workstation Edition")
        self.resize(1100, 720)

        # Persistent settings
        self.settings_path = os.path.join(".ui_state", "settings.json")
        self.state = _read_json(self.settings_path, {})

        # Central widget & tabs
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()
        v.addWidget(self.tabs)

        # --- Tabs
        self.tab_synth = self._build_synthesis_tab()
        self.tab_dataset = self._build_dataset_tab()
        self.tab_training = self._build_training_tab()
        self.tab_adapt = self._build_adaptation_tab()
        self.tab_transcripts = self._build_transcripts_tab()
        self.tabs.addTab(self.tab_synth, "Synthesis")
        self.tabs.addTab(self.tab_dataset, "Dataset")
        self.tabs.addTab(self.tab_training, "Training")
        self.tabs.addTab(self.tab_adapt, "Adaptation")
        self.tabs.addTab(self.tab_transcripts, "Transcripts")
        # Suppose you have self.tabs: QTabWidget, with indexes for each tab
        self.tabs.currentChanged.connect(self._on_tab_changed)


        # Bottom status area
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QtWidgets.QLabel("Ready.")
        v.addWidget(self.progress)
        self.console_output = QtWidgets.QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.console_output.setWordWrapMode(QtGui.QTextOption.WrapAnywhere)
        self.console_output.setMaximumHeight(80)  # Fixed height to prevent expansion
        self.console_output.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.console_output.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.console_output.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        v.addWidget(self.console_output)

        # Device banner
        self._print_device_banner()

        # restore small bits
        self._restore_ui_state()
    
    def _join_cleaners(self, cleaners):
        # NVIDIA hparams expects a python-like list repr or a pipe/comma list.
        # We’ll use a pipe so commas in text are safe.
        if isinstance(cleaners, (list, tuple)):
            return "|".join(str(c).strip() for c in cleaners if str(c).strip())
        return str(cleaners).strip()
        
    def _show_help(self, topic: str):
        import textwrap
        html_map = {
            "synthesis": textwrap.dedent("""
                <h3>Synthesis – What the settings do</h3>
                <ul>
                  <li><b>Tacotron2 checkpoint</b>: The text→mel model. Must match your cleaners & symbol set.</li>
                  <li><b>WaveGlow checkpoint</b>: The mel→audio vocoder. If empty, app falls back to Griffin–Lim (lower quality).</li>
                  <li><b>Device</b>: <i>auto</i> picks CUDA when available, otherwise CPU. CUDA is much faster.</li>
                  <li><b>WaveGlow sigma</b>: Sampling temperature (0.50–1.20). Lower = cleaner/less expressive; higher = livelier/noisier. Typical 0.7–0.9.</li>
                  <li><b>Denoiser strength</b>: Removes faint hiss after WaveGlow. 0.010–0.020 is subtle; too high can dull audio.</li>
                  <li><b>Text</b>: Cleaned using your chosen <i>text cleaners</i> (see Training tab). If cleaners strip everything, T2 returns empty.</li>
                </ul>
                <p><i>Tip:</i> If audio clips, we peak-normalize to −3 dBFS.</p>
            """),
            "training": textwrap.dedent("""
                <h3>Training – How each option affects Tacotron2</h3>
                <ul>
                  <li><b>Train/Val filelist</b>: Lines of <code>/abs/path.wav|normalized text</code>. Use absolute paths. Val ~2% of data.</li>
                  <li><b>Sampling rate</b> (default 22050): Must match your audio & mel configs; changing it invalidates old checkpoints.</li>
                  <li><b>Hop / Win / FFT</b>: STFT params for mel extraction. Stock: 256 / 1024 / 1024 at 22.05 kHz.</li>
                  <li><b># Mel channels</b>: Usually 80. Changing requires retraining from scratch (mel dimension is baked into weights).</li>
                  <li><b>Batch size</b>: Larger = faster, but more VRAM. If you OOM, reduce this. Mixed precision can help.</li>
                  <li><b>Epochs</b>: Upper bound on training sweeps; convergence is data-dependent.</li>
                  <li><b>Iters per checkpoint</b>: How often to save <code>tacotron2_*.pt</code>. Smaller = more frequent snapshots.</li>
                  <li><b>Learning rate</b>: Start at 1e-3. If training diverges (loss explodes), reduce (e.g., 5e-4).</li>
                  <li><b>Weight decay</b>: L2 regularization; small values (1e-6) help generalization without over-penalizing.</li>
                  <li><b>Grad clip thresh</b>: Caps gradient norm to stabilize training; 1.0 is standard.</li>
                  <li><b>FP16</b> (native amp): Speeds up & saves VRAM on CUDA. Disable if you see NaNs or instability.</li>
                  <li><b>cudnn.benchmark</b>: Lets cuDNN choose fastest kernels for fixed shapes. Enable for speed, disable for fully dynamic lengths.</li>
                  <li><b># GPUs / Distributed</b>: Multi-GPU training via DDP. Keep off unless you run the official distributed script entry.</li>
                  <li><b>Text cleaners</b>: Preprocessor for text→IDs. Must match what you’ll synthesize with. Typical: <code>english_cleaners</code>.</li>
                </ul>
                <p><i>Important:</i> Any change to SR / hop / win / FFT / mel channels requires re-generating mels and retraining.</p>
            """),
            "adaptation": textwrap.dedent("""
                <h3>Adaptation – Fine-tuning from a teacher checkpoint</h3>
                <ul>
                  <li><b>Enable adaptation</b>: When ON, training loads <i>warm-start</i> weights and optionally freezes parts.</li>
                  <li><b>Warm-start checkpoint</b>: A trained Tacotron2 <code>.pt</code>. Should be architecturally compatible (same mel dims, etc.).</li>
                  <li><b>Freeze encoder</b>: Keeps the text side stable; good when adapting to a new voice from similar text domain.</li>
                  <li><b>Freeze postnet</b>: Prevents postnet from drifting; can keep mels crisp while the decoder adapts.</li>
                  <li><b>LR multiplier</b>: Scales LR during adaptation. &lt; 1.0 for gentle fine-tune (e.g., 0.2–0.5).</li>
                  <li><b>Transfer layers</b>: Comma-list of module prefixes to copy/emphasize (e.g., <code>embedding,encoder</code>). Leave empty for vanilla warm-start.</li>
                </ul>
                <p><i>Guideline:</i> Start with freezing encoder, postnet unfrozen, LR mult 0.5. Unfreeze progressively if underfitting.</p>
            """),
        }
        html = html_map.get(topic, "<h3>No help available.</h3>")

        # Use a scrollable dialog for readability
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Help — {topic.capitalize()}")
        dlg.resize(600, 500)

        layout = QtWidgets.QVBoxLayout(dlg)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setHtml(html)
        layout.addWidget(view)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.exec_()


    def _on_tab_changed(self, idx):
        label = self.tabs.tabText(idx).lower()
        if "training" in label:
            self._training_load_from_cfg()
        elif "adapt" in label:
            self._adaptation_load_from_cfg()

    def _training_apply_save(self):
        cfg = load_config()
        cfg["data"]["training-files"] = self.tr_train_files.text().strip()
        cfg["data"]["validation-files"] = self.tr_val_files.text().strip()
        cfg["audio"]["sampling-rate"] = int(self.tr_sr.value())
        cfg["audio"]["hop-length"] = int(self.tr_hop.value())
        cfg["audio"]["win-length"] = int(self.tr_win.value())
        cfg["audio"]["filter-length"] = int(self.tr_fft.value())
        cfg["model"]["n-mel-channels"] = int(self.tr_mels.value())
        cfg["data"]["batch-size"] = int(self.tr_bs.value())
        cfg["data"]["epochs"] = int(self.tr_epochs.value())
        cfg["data"]["iters-per-checkpoint"] = int(self.tr_ipc.value())
        cfg["optimization"]["learning-rate"] = float(self.tr_lr.value())
        cfg["optimization"]["weight-decay"] = float(self.tr_wd.value())
        cfg["optimization"]["grad-clip-thresh"] = float(self.tr_clip.value())
        cfg["optimization"]["fp16-run"] = bool(self.tr_fp16.isChecked())
        cfg["optimization"]["cudnn-benchmark"] = bool(self.tr_cudnn_bench.isChecked())
        cfg["distributed"]["n-gpus"] = int(self.tr_ngpus.value())
        cfg["distributed"]["distributed-run"] = bool(self.tr_dist.isChecked())
        cleaners = [c.strip() for c in self.tr_cleaners.text().split(",") if c.strip()]
        cfg["data"]["text-cleaners"] = cleaners or ["english_cleaners"]
        save_config(cfg)
        self.tr_status.setText("Saved to config.json")
                    
    def _training_load_from_cfg(self):
        cfg = load_config()
        # paths
        self.tr_train_files.setText(cfg["data"]["training-files"])
        self.tr_val_files.setText(cfg["data"]["validation-files"])
        # audio
        self.tr_sr.setValue(int(cfg["audio"]["sampling-rate"]))
        self.tr_hop.setValue(int(cfg["audio"]["hop-length"]))
        self.tr_win.setValue(int(cfg["audio"]["win-length"]))
        self.tr_fft.setValue(int(cfg["audio"]["filter-length"]))
        self.tr_mels.setValue(int(cfg["model"]["n-mel-channels"]))
        # train/optim
        self.tr_bs.setValue(int(cfg["data"]["batch-size"]))
        self.tr_epochs.setValue(int(cfg["data"]["epochs"]))
        self.tr_ipc.setValue(int(cfg["data"]["iters-per-checkpoint"]))
        self.tr_lr.setValue(float(cfg["optimization"]["learning-rate"]))
        self.tr_wd.setValue(float(cfg["optimization"]["weight-decay"]))
        self.tr_clip.setValue(float(cfg["optimization"]["grad-clip-thresh"]))
        self.tr_fp16.setChecked(bool(cfg["optimization"]["fp16-run"]))
        self.tr_cudnn_bench.setChecked(bool(cfg["optimization"]["cudnn-benchmark"]))
        # distributed
        self.tr_ngpus.setValue(int(cfg["distributed"]["n-gpus"]))
        self.tr_dist.setChecked(bool(cfg["distributed"]["distributed-run"]))
        # cleaners
        cleaners = cfg["data"]["text-cleaners"]
        self.tr_cleaners.setText(",".join(cleaners) if isinstance(cleaners, list) else str(cleaners))
        self.tr_status.setText("Loaded from config.json")

    def _adaptation_load_from_cfg(self):
        cfg = load_config()
        ad = cfg.get("adaptation", {})
        self.ad_enable.setChecked(bool(ad.get("enable", False)))
        self.ad_warm.setText(ad.get("warm-start", ""))
        self.ad_freeze_enc.setChecked(bool(ad.get("freeze-encoder", False)))
        self.ad_freeze_post.setChecked(bool(ad.get("freeze-postnet", False)))
        self.ad_lr_mult.setValue(float(ad.get("lr-mult", 1.0)))
        tr_layers = ad.get("transfer-layers", [])
        self.ad_layers.setText(",".join(tr_layers) if isinstance(tr_layers, list) else str(tr_layers))
        self.ad_status.setText("Loaded from config.json")

    def _adaptation_apply_save(self):
        cfg = load_config()
        cfg["adaptation"]["enable"] = bool(self.ad_enable.isChecked())
        cfg["adaptation"]["warm-start"] = self.ad_warm.text().strip()
        cfg["adaptation"]["freeze-encoder"] = bool(self.ad_freeze_enc.isChecked())
        cfg["adaptation"]["freeze-postnet"] = bool(self.ad_freeze_post.isChecked())
        cfg["adaptation"]["lr-mult"] = float(self.ad_lr_mult.value())
        layers = [s.strip() for s in self.ad_layers.text().split(",") if s.strip()]
        cfg["adaptation"]["transfer-layers"] = layers
        save_config(cfg)
        self.ad_status.setText("Saved to config.json")

    def on_generate_adaptation_pairs(self):
        # Resolve dataset root (prefer live banner state, else textbox)
        ds_root = getattr(self, "dataset_root", None) or self.ds_dir_edit.text().strip()
        if not ds_root:
            QtWidgets.QMessageBox.warning(self, "Dataset", "Select a dataset folder first.")
            return

        ds = Path(ds_root)
        meta = ds / "metadata.csv"
        wavs = ds / "wavs"
        adapt = ds / "adaptation"

        if not meta.exists():
            QtWidgets.QMessageBox.warning(self, "Dataset", f"metadata.csv not found at:\n{meta}")
            return
        if not wavs.is_dir():
            QtWidgets.QMessageBox.warning(self, "Dataset", f"'wavs' folder not found at:\n{wavs}")
            return

        adapt.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = 0
        skipped = 0
        errors = 0

        try:
            with meta.open("r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    name = row[0].strip()
                    text = row[1].strip()

                    # Accept either bare name or name with .wav suffix in metadata
                    base = name[:-4] if name.lower().endswith(".wav") else name
                    src_wav = wavs / f"{base}.wav"
                    if not src_wav.exists():
                        missing += 1
                        continue

                    dst_wav = adapt / f"{base}.wav"
                    dst_txt = adapt / f"{base}.txt"

                    try:
                        shutil.copy2(src_wav, dst_wav)
                        with dst_txt.open("w", encoding="utf-8") as tf:
                            tf.write(text + "\n")
                        copied += 1
                    except Exception:
                        errors += 1
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Adaptation export", f"Failed: {e}")
            self.ds_status.setText("Adaptation export failed.")
            return

        msg = f"Adaptation export complete: {copied} files, {missing} missing wavs, {errors} errors."
        self.ds_status.setText(msg)
        QtWidgets.QMessageBox.information(self, "Adaptation export", msg)


    # ------------------------------ UI builders ------------------------------

    def _build_synthesis_tab(self):
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)

        row = 0
        self.taco_path = QtWidgets.QLineEdit(self.state.get("taco_path", autodiscover_checkpoint() or ""))
        btn_taco = QtWidgets.QPushButton("Browse…")
        btn_taco.clicked.connect(lambda: self._pick_file(self.taco_path))
        g.addWidget(QtWidgets.QLabel("Tacotron2 Checkpoint"), row, 0)
        g.addWidget(self.taco_path, row, 1)
        g.addWidget(btn_taco, row, 2)

        row += 1
        self.wg_path = QtWidgets.QLineEdit(self.state.get("wg_path", ""))
        btn_wg = QtWidgets.QPushButton("Browse…")
        btn_wg.clicked.connect(lambda: self._pick_file(self.wg_path))
        g.addWidget(QtWidgets.QLabel("WaveGlow Checkpoint (optional)"), row, 0)
        g.addWidget(self.wg_path, row, 1)
        g.addWidget(btn_wg, row, 2)

        row += 1
        self.device_combo = QtWidgets.QComboBox()
        devs = ["cpu"]
        if torch.cuda.is_available():
            devs.append("cuda")
        self.device_combo.addItems(devs)
        self.device_combo.setCurrentText(self.state.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        g.addWidget(QtWidgets.QLabel("Device"), row, 0)
        g.addWidget(self.device_combo, row, 1)

        row += 1
        self.sigma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sigma_slider.setToolTip("WaveGlow temperature. Controls expressiveness. 0.5–0.8 typical.")
        self.sigma_slider.setMinimum(1)
        self.sigma_slider.setMaximum(100)
        self.sigma_slider.setValue(int(self.state.get("sigma", 60)))
        self.sigma_label = QtWidgets.QLabel("Sigma: 0.60")
        self.sigma_slider.valueChanged.connect(lambda v: self.sigma_label.setText(f"Sigma: {v/100:.2f}"))
        g.addWidget(self.sigma_label, row, 0)
        g.addWidget(self.sigma_slider, row, 1)

        row += 1
        self.denoise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.denoise_slider.setToolTip("Post-vocoder hiss removal. 0.010–0.020 recommended range.")

        self.denoise_slider.setMinimum(0)
        self.denoise_slider.setMaximum(100)
        self.denoise_slider.setValue(int(self.state.get("denoise", 0)))
        self.denoise_label = QtWidgets.QLabel("Denoise: 0.00")
        self.denoise_slider.valueChanged.connect(lambda v: self.denoise_label.setText(f"Denoise: {v/100:.2f}"))
        g.addWidget(self.denoise_label, row, 0)
        g.addWidget(self.denoise_slider, row, 1)

        row += 1
        self.text_edit = QtWidgets.QPlainTextEdit(self.state.get("last_text", "Hello world. This is a test."))
        g.addWidget(QtWidgets.QLabel("Text"), row, 0)
        g.addWidget(self.text_edit, row, 1, 1, 2)

        row += 1
        synth = QtWidgets.QPushButton("Generate Speech")
        play = QtWidgets.QPushButton("Play")
        save = QtWidgets.QPushButton("Save Wave")
        self.synth_help = QtWidgets.QPushButton("Help")
        synth.clicked.connect(self._on_synthesize)
        play.clicked.connect(self._on_play)
        save.clicked.connect(self._on_save_as)
        g.addWidget(synth, row, 0)
        g.addWidget(play, row, 1)
        g.addWidget(save, row, 2)
        g.addWidget(self.synth_help, row, 3)  # add this

        # Media player
        self.player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.LowLatency)

        return w

    def _build_dataset_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        paths = dataset_paths()
        self.dataset_root, self.wavs_dir, self.metadata_csv = paths

        # Path banner
        self.dataset_label = QtWidgets.QLabel(f"Active dataset: {self.dataset_root}")
        v.addWidget(self.dataset_label)

        # Existing controls (Record / Upload...)
        h = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_upload_audio = QtWidgets.QPushButton("Upload Audio Files…")
        self.btn_upload_txt = QtWidgets.QPushButton("Upload Transcript (.txt)…")
        self.btn_open_logs = QtWidgets.QPushButton("Open Logs Folder")
        self.btn_open_logs.setToolTip(f"Log file: {LOG_PATH}")
        self.btn_open_logs.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(LOG_DIR))
        )
        h.addWidget(self.btn_record)
        h.addWidget(self.btn_stop)
        h.addWidget(self.btn_upload_audio)
        h.addWidget(self.btn_upload_txt)
        h.addWidget(self.btn_open_logs); h.addStretch(1)
        v.addLayout(h)

        # List of wavs
        self.list_audio = QtWidgets.QListWidget()
        v.addWidget(self.list_audio)

        # --- NEW: Auto-transcribe & filelists group ---
        group = QtWidgets.QGroupBox("Auto-transcribe & build Tacotron2 filelists")
        form = QtWidgets.QFormLayout(group)

        # default to current dataset root
        self.ds_dir_edit  = QtWidgets.QLineEdit(self.dataset_root)
        self.ds_dir_btn   = QtWidgets.QPushButton("Browse…")
        self.ds_resample  = QtWidgets.QCheckBox("Resample to 22,050 Hz")
        self.ds_resample.setChecked(True)
        self.ds_run_btn   = QtWidgets.QPushButton("Generate transcripts && filelists")
        self.ds_adapt_btn = QtWidgets.QPushButton("Generate adaptation wav/txt")
        self.ds_adapt_btn.setToolTip("Copy wavs and create .txt transcripts into the dataset's 'adaptation/' folder from metadata.csv")
        self.ds_status    = QtWidgets.QLabel("")

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.ds_dir_edit)
        row.addWidget(self.ds_dir_btn)
        form.addRow("Dataset root:", row)
        form.addRow("", self.ds_resample)
        form.addRow("", self.ds_run_btn)
        form.addRow("", self.ds_adapt_btn)
        
        form.addRow("Status:", self.ds_status)

        v.addWidget(group)

        # Wire
        self.ds_adapt_btn.clicked.connect(self.on_generate_adaptation_pairs)
        self.btn_record.clicked.connect(self._start_record)
        self.btn_stop.clicked.connect(self._stop_record)
        self.btn_upload_audio.clicked.connect(self._upload_audio_files)
        self.btn_upload_txt.clicked.connect(self._upload_transcript)

        def _pick_ds_dir():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select dataset folder (contains wavs/)")
            if d:
                self.ds_dir_edit.setText(d)
                # live update banner, wavs dir, etc.
                self.dataset_root = d
                self.wavs_dir = os.path.join(d, "wavs")
                self.metadata_csv = os.path.join(d, "metadata.csv")
                self.dataset_label.setText(f"Active dataset: {self.dataset_root}")
                self._refresh_audio_list()

        self.ds_dir_btn.clicked.connect(_pick_ds_dir)
        self.ds_run_btn.clicked.connect(self.on_dataset_generate)

        self._refresh_audio_list()
        return w


    def _build_training_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        form = QtWidgets.QFormLayout()

        # Data paths
        self.tr_train_files = QtWidgets.QLineEdit("filelists/myset_train.txt")
        self.tr_val_files   = QtWidgets.QLineEdit("filelists/myset_val.txt")
        self.tr_resume_ckpt = QtWidgets.QLineEdit("")
        btn_train = QtWidgets.QPushButton("Browse…")
        btn_val   = QtWidgets.QPushButton("Browse…")
        btn_resume_browse = QtWidgets.QPushButton("Browse…")

        def _pick_train():
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select train filelist", ROOT_DIR, "Text (*.txt);;All (*)")
            if p: self.tr_train_files.setText(p)
        def _pick_val():
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select val filelist", ROOT_DIR, "Text (*.txt);;All (*)")
            if p: self.tr_val_files.setText(p)
        btn_train.clicked.connect(_pick_train); btn_val.clicked.connect(_pick_val)
        def _pick_resume():
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select checkpoint to resume", ROOT_DIR,
                                                         "PyTorch (*.pt *.pth);;All (*)")
            if p:
                self.tr_resume_ckpt.setText(p)
        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(self.tr_train_files); row1.addWidget(btn_train)
        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(self.tr_val_files);   row2.addWidget(btn_val)
        rowr = QtWidgets.QHBoxLayout()
        rowr.addWidget(self.tr_resume_ckpt)
        rowr.addWidget(btn_resume_browse)
        form.addRow("Train filelist:", row1)
        form.addRow("Val filelist:",   row2)
        form.addRow("Resume from checkpoint:", rowr)
        btn_resume_browse.clicked.connect(_pick_resume)

        # Audio
        self.tr_sr   = QtWidgets.QSpinBox(); self.tr_sr.setRange(8000, 96000)
        self.tr_hop  = QtWidgets.QSpinBox(); self.tr_hop.setRange(64, 4096)
        self.tr_win  = QtWidgets.QSpinBox(); self.tr_win.setRange(128, 8192)
        self.tr_fft  = QtWidgets.QSpinBox(); self.tr_fft.setRange(128, 8192)
        self.tr_mels = QtWidgets.QSpinBox(); self.tr_mels.setRange(20, 256)
        form.addRow("Sampling rate:", self.tr_sr)
        form.addRow("Hop length:",    self.tr_hop)
        form.addRow("Win length:",    self.tr_win)
        form.addRow("FFT size:",      self.tr_fft)
        form.addRow("# Mel channels:",self.tr_mels)

        # Training / optimization (maps to your old opt_*)
        self.tr_bs       = QtWidgets.QSpinBox();        self.tr_bs.setRange(1, 512);     self.tr_bs.setValue(32)
        self.opt_workers = QtWidgets.QSpinBox();        self.opt_workers.setRange(0, 16); self.opt_workers.setValue(0 if sys.platform.startswith("win") else 4)
        self.opt_lr      = QtWidgets.QDoubleSpinBox();  self.opt_lr.setDecimals(6); self.opt_lr.setRange(1e-6, 1.0); self.opt_lr.setSingleStep(1e-4); self.opt_lr.setValue(1e-3)
        self.tr_epochs   = QtWidgets.QSpinBox();        self.tr_epochs.setRange(1, 5000); self.tr_epochs.setValue(1500)
        self.tr_ipc      = QtWidgets.QSpinBox();        self.tr_ipc.setRange(10, 100000); self.tr_ipc.setValue(1000)
        self.tr_wd       = QtWidgets.QDoubleSpinBox();  self.tr_wd.setDecimals(8); self.tr_wd.setRange(0.0, 1e-2); self.tr_wd.setSingleStep(1e-6); self.tr_wd.setValue(1e-6)
        self.tr_clip     = QtWidgets.QDoubleSpinBox();  self.tr_clip.setDecimals(2); self.tr_clip.setRange(0.0, 100.0); self.tr_clip.setSingleStep(0.1); self.tr_clip.setValue(1.0)
        self.tr_lr       = self.opt_lr  # keep both names consistent

        self.tr_fp16 = QtWidgets.QCheckBox("Enable FP16 (native amp)")
        self.tr_cudnn_bench = QtWidgets.QCheckBox("cudnn.benchmark")  # replaces tr_cudnn_bench

        form.addRow("Batch size:",              self.tr_bs)
        form.addRow("Dataloader workers:",      self.opt_workers)
        form.addRow("Learning rate:",           self.opt_lr)
        form.addRow("Epochs:",                  self.tr_epochs)
        form.addRow("Iters per checkpoint:",    self.tr_ipc)
        form.addRow("Weight decay:",            self.tr_wd)
        form.addRow("Grad clip thresh:",        self.tr_clip)
        form.addRow("",                         self.tr_fp16)
        form.addRow("",                         self.tr_cudnn_bench)

        # Distributed
        self.tr_ngpus = QtWidgets.QSpinBox(); self.tr_ngpus.setRange(1, 8)
        self.tr_dist  = QtWidgets.QCheckBox("Distributed run (DDP)")
        form.addRow("# GPUs:", self.tr_ngpus)
        form.addRow("", self.tr_dist)

        # Cleaners
        self.tr_cleaners = QtWidgets.QLineEdit("english_cleaners")
        form.addRow("Text cleaners (comma-sep):", self.tr_cleaners)

        # Output directory for checkpoints (provides train_out_dir used in save/load)
        self.train_out_dir = QtWidgets.QLineEdit(self.state.get("train_out_dir", os.path.join("checkpoints", "runs", "run1")))
        btn_out = QtWidgets.QPushButton("Browse…")
        def _pick_out():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", ROOT_DIR)
            if d: self.train_out_dir.setText(d)
        btn_out.clicked.connect(_pick_out)
        rowo = QtWidgets.QHBoxLayout(); rowo.addWidget(self.train_out_dir); rowo.addWidget(btn_out)
        form.addRow("Output directory:", rowo)

        # Tooltips
        self.tr_sr.setToolTip("Audio sampling rate. Must match your .wav files and mel extraction. Default 22050 Hz.")
        self.tr_hop.setToolTip("STFT hop length (samples). With 22050 Hz, 256 ≈ 11.6 ms frames.")
        self.tr_win.setToolTip("STFT window length (samples). Usually equal to FFT size; default 1024.")
        self.tr_fft.setToolTip("STFT FFT size. Larger = finer freq resolution; must match during inference.")
        self.tr_mels.setToolTip("Mel filterbank channels. 80 typical; changing requires retraining.")
        self.tr_bs.setToolTip("Batch size per iteration. Larger = faster but more VRAM; reduce if you OOM.")
        self.opt_workers.setToolTip("Dataloader workers. On Windows, keep 0.")
        self.opt_lr.setToolTip("Adam learning rate. Start 1e-3; if loss explodes or NaNs, try 5e-4.")
        self.tr_epochs.setToolTip("Max training epochs. Convergence depends on dataset size/quality.")
        self.tr_ipc.setToolTip("Iterations per checkpoint. Lower = more frequent .pt saves.")
        self.tr_wd.setToolTip("Weight decay (L2). Small values like 1e-6 can improve generalization.")
        self.tr_clip.setToolTip("Gradient clipping threshold (global norm). 1.0 is a stable default.")
        self.tr_fp16.setToolTip("Enable mixed precision on CUDA for speed/VRAM. Disable if you see NaNs/instability.")
        self.tr_cudnn_bench.setToolTip("Let cuDNN auto-tune kernels (faster with fixed shapes).")

        v.addLayout(form)

        # Apply / Load
        btns = QtWidgets.QHBoxLayout()
        self.tr_load_btn  = QtWidgets.QPushButton("Load from config.json")
        self.tr_apply_btn = QtWidgets.QPushButton("Apply && Save")
        self.tr_help_btn  = QtWidgets.QPushButton("Help")
        
        btns.addWidget(self.tr_load_btn)
        btns.addWidget(self.tr_apply_btn)
        btns.addStretch(1)
        btns.addWidget(self.tr_help_btn)
        v.addLayout(btns)
        run_row = QtWidgets.QHBoxLayout()
        self.tr_start_btn = QtWidgets.QPushButton("Start Training")
        self.tr_stop_btn  = QtWidgets.QPushButton("Stop")
        self.tr_stop_btn.setEnabled(False)
        run_row.addWidget(self.tr_start_btn)
        run_row.addWidget(self.tr_stop_btn)
        run_row.addStretch(1)
        v.addLayout(run_row)

        self.tr_start_btn.clicked.connect(self._on_start_training)
        self.tr_stop_btn.clicked.connect(self._on_stop_training)

        self.tr_status = QtWidgets.QLabel("")
        v.addWidget(self.tr_status)

        # Wire
        self.tr_load_btn.clicked.connect(self._training_load_from_cfg)
        self.tr_apply_btn.clicked.connect(self._training_apply_save)
        self.tr_help_btn.clicked.connect(lambda: self._show_help("training"))

        return w

    ### CHANGED: this is the function that actually builds the command-line hparams.
    ### we inject NVIDIA-style model fields here, pulled from config.json if present.

def _build_train_cmd(self, warm_path: str = ""):
    cfg = load_config()

    t2_dir   = os.path.join(ROOT_DIR, "third_party", "tacotron2")
    train_py = os.path.join(t2_dir, "train.py")

    # Resolve/ensure out dirs
    out_dir = os.path.abspath(self.train_out_dir.text().strip())
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- helpers ---
    def _bool_str(x: bool) -> str:
        # NVIDIA hparam parser expects capitalized True/False
        return "True" if bool(x) else "False"

    # Filelists MUST be passed as dedicated flags (not via --hparams)
    training_files   = os.path.abspath(cfg["data"]["training-files"])
    validation_files = os.path.abspath(cfg["data"]["validation-files"])

    # list-literal fields must be Python-literal strings
    cleaners_lit      = _as_list_lit(cfg["data"]["text-cleaners"])
    ignore_layers_lit = _as_list_lit(cfg.get("model", {}).get("ignore-layers", ["embedding.weight"]))

    mcfg = cfg.get("model", {})

    # ---------------- Build hparams ----------------
    hp = {
        # audio
        "max_wav_value":         float(cfg["audio"]["max-wav-value"]),
        "sampling_rate":         int(cfg["audio"]["sampling-rate"]),
        "filter_length":         int(cfg["audio"]["filter-length"]),
        "hop_length":            int(cfg["audio"]["hop-length"]),
        "win_length":            int(cfg["audio"]["win-length"]),
        "mel_fmin":              float(cfg["audio"]["mel-fmin"]),
        "mel_fmax":              float(cfg["audio"]["mel-fmax"]),

        # core model (explicitly mirror NVIDIA names)
        "n_mel_channels":        int(mcfg["n-mel-channels"]),
        "n_symbols":             int(mcfg.get("n-symbols", 148)),
        "symbols_embedding_dim": int(mcfg["symbols-embedding-dim"]),
        "encoder_kernel_size":   int(mcfg["encoder-kernel-size"]),
        "encoder_n_convolutions":int(mcfg["encoder-n-convolutions"]),
        "encoder_embedding_dim": int(mcfg["encoder-embedding-dim"]),
        "attention_rnn_dim":     int(mcfg["attention-rnn-dim"]),
        "attention_dim":         int(mcfg["attention-dim"]),
        "attention_location_n_filters":  int(mcfg["attention-location-n-filters"]),
        "attention_location_kernel_size":int(mcfg["attention-location-kernel-size"]),
        "n_frames_per_step":     int(mcfg["n-frames-per-step"]),
        "decoder_rnn_dim":       int(mcfg["decoder-rnn-dim"]),
        "prenet_dim":            int(mcfg["prenet-dim"]),
        "max_decoder_steps":     int(mcfg["max-decoder-steps"]),
        "gate_threshold":        float(mcfg["gate-threshold"]),
        "p_attention_dropout":   float(mcfg["p-attention-dropout"]),
        "p_decoder_dropout":     float(mcfg["p-decoder-dropout"]),
        "postnet_embedding_dim": int(mcfg["postnet-embedding-dim"]),
        "postnet_kernel_size":   int(mcfg["postnet-kernel-size"]),
        "postnet_n_convolutions":int(mcfg["postnet-n-convolutions"]),
        "decoder_no_early_stopping": _bool_str(mcfg["decoder-no-early-stopping"]),
        "mask_padding":          _bool_str(mcfg["mask-padding"]),

        # text & ignore list as python literals
        "text_cleaners":         cleaners_lit,
        "ignore_layers":         ignore_layers_lit,

        # optimization/runtime
        "batch_size":            int(cfg["data"]["batch-size"]),
        "epochs":                int(cfg["data"]["epochs"]),
        "iters_per_checkpoint":  int(cfg["data"]["iters-per-checkpoint"]),
        "learning_rate":         float(cfg["optimization"]["learning-rate"]),
        "weight_decay":          float(cfg["optimization"]["weight-decay"]),
        "grad_clip_thresh":      float(cfg["optimization"]["grad-clip-thresh"]),
        "fp16_run":              _bool_str(cfg["optimization"]["fp16-run"]),
        "cudnn_benchmark":       _bool_str(cfg["optimization"]["cudnn-benchmark"]),
        # force this if you want to override NVIDIA default True
        "cudnn_enabled":         _bool_str(cfg["optimization"].get("cudnn-enabled", True)),
    }

    # Make the comma-separated --hparams string (k=v). Values that are already
    # python-literal strings (e.g. "['english_cleaners']") are safe to pass verbatim.
    hparams_parts = [f"{k}={v}" for k, v in hp.items()]
    hparams_str = ",".join(hparams_parts)

    # ---------------- Base command (single GPU) ----------------
    cmd = [
        sys.executable, train_py,
        "--output_directory", out_dir,
        "--log_directory",    log_dir,
        "--hparams",          hparams_str,
        # CRITICAL: filelists as real flags (not inside --hparams)
        "--training_files",   training_files,
        "--validation_files", validation_files,
    ]

    # Resume / Warm-start
    resume_ckpt = self.tr_resume_ckpt.text().strip() if hasattr(self, "tr_resume_ckpt") else ""
    if resume_ckpt:
        cmd += ["--checkpoint_path", os.path.abspath(resume_ckpt)]
    elif warm_path:
        cmd += ["--checkpoint_path", os.path.abspath(warm_path), "--warm_start"]

    # ---------------- Distributed (DDP) override ----------------
    if hasattr(self, "tr_distributed") and self.tr_distributed.isChecked():
        n_gpus = int(cfg["distributed"]["n-gpus"])
        launcher = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            "--master_addr=127.0.0.1",
            "--master_port=29500",
        ]
        ddp_cmd = launcher + [
            train_py,
            "--output_directory", out_dir,
            "--log_directory",    log_dir,
            "--hparams",          hparams_str,
            "--training_files",   training_files,
            "--validation_files", validation_files,
        ]
        if resume_ckpt:
            ddp_cmd += ["--checkpoint_path", os.path.abspath(resume_ckpt)]
        elif warm_path:
            ddp_cmd += ["--checkpoint_path", os.path.abspath(warm_path), "--warm_start"]
        cmd = ddp_cmd  # replace base cmd

    # Log the exact command for sanity
    print("[train cmd]", " ".join(cmd))

    return cmd, t2_dir


    def _on_start_training(self):
        cmd, cwd = self._build_train_cmd(warm_path="")
        self.tr_status.setText("Starting training…")
        self.tr_thread = QtCore.QThread()
        self.tr_worker = TrainWorker(cmd, cwd=cwd)
        self.tr_worker.moveToThread(self.tr_thread)
        self.tr_thread.started.connect(self.tr_worker.run)
        self.tr_worker.progress.connect(self._append_train_log)
        self.tr_worker.finished.connect(self._train_done)
        self.tr_worker.failed.connect(self._train_failed)
        # cleanup
        self.tr_worker.finished.connect(self.tr_thread.quit)
        self.tr_worker.failed.connect(self.tr_thread.quit)
        self.tr_worker.finished.connect(self.tr_worker.deleteLater)
        self.tr_worker.failed.connect(self.tr_worker.deleteLater)
        self.tr_thread.finished.connect(self.tr_thread.deleteLater)
        self.tr_thread.start()
        self.tr_start_btn.setEnabled(False)
        self.tr_stop_btn.setEnabled(True)

    def _on_stop_training(self):
        if hasattr(self, "tr_worker"):
            self.tr_worker.stop()

    def _on_start_adaptation(self):
        warm = self.ad_warm.text().strip()
        if not warm or not os.path.isfile(warm):
            QtWidgets.QMessageBox.warning(self, "Adaptation", "Select a valid warm-start checkpoint (.pt/.pth).")
            return
        cmd, cwd = self._build_train_cmd(warm_path=warm)
        self.ad_status.setText("Starting adaptation…")
        self.ad_thread = QtCore.QThread()
        self.ad_worker = TrainWorker(cmd, cwd=cwd)
        self.ad_worker.moveToThread(self.ad_thread)
        self.ad_thread.started.connect(self.ad_worker.run)
        self.ad_worker.progress.connect(self._append_adapt_log)
        self.ad_worker.finished.connect(self._adapt_done)
        self.ad_worker.failed.connect(self._adapt_failed)
        # cleanup
        self.ad_worker.finished.connect(self.ad_thread.quit)
        self.ad_worker.failed.connect(self.ad_thread.quit)
        self.ad_worker.finished.connect(self.ad_worker.deleteLater)
        self.ad_worker.failed.connect(self.ad_worker.deleteLater)
        self.ad_thread.finished.connect(self.ad_thread.deleteLater)
        self.ad_thread.start()
        self.ad_start_btn.setEnabled(False)
        self.ad_stop_btn.setEnabled(True)

    def _on_stop_adaptation(self):
        if hasattr(self, "ad_worker"):
            self.ad_worker.stop()

    # tiny helpers to route logs to your status/console
    def _append_train_log(self, line:str):
        self.tr_status.setText(line)

    def _append_adapt_log(self, line:str):
        self.ad_status.setText(line)

    def _train_done(self, msg:str):
        self.tr_status.setText(msg)
        self.tr_start_btn.setEnabled(True)
        self.tr_stop_btn.setEnabled(False)

    def _train_failed(self, msg:str):
        self.tr_status.setText("Failed: " + msg)
        self.tr_start_btn.setEnabled(True)
        self.tr_stop_btn.setEnabled(False)

    def _adapt_done(self, msg:str):
        self.ad_status.setText(msg)
        self.ad_start_btn.setEnabled(True)
        self.ad_stop_btn.setEnabled(False)

    def _adapt_failed(self, msg:str):
        self.ad_status.setText("Failed: " + msg)
        self.ad_start_btn.setEnabled(True)
        self.ad_stop_btn.setEnabled(False)


    def _build_adaptation_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        form = QtWidgets.QFormLayout()

        # Master toggle
        self.ad_enable = QtWidgets.QCheckBox("Enable adaptation (fine-tune from teacher)")
        form.addRow(self.ad_enable)

        # Warm start checkpoint
        self.ad_warm = QtWidgets.QLineEdit("")
        btn_warm = QtWidgets.QPushButton("Browse…")
        def _pick_warm():
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select teacher Tacotron2 checkpoint", ROOT_DIR,
                "PyTorch (*.pt *.pth);;All (*)"
            )
            if p: self.ad_warm.setText(p)
        btn_warm.clicked.connect(_pick_warm)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.ad_warm); row.addWidget(btn_warm)
        form.addRow("Warm-start checkpoint:", row)

        # Adaptation folder (.wav + .txt pairs)
        self.adapt_folder = QtWidgets.QLineEdit(self.state.get("adapt_folder", ""))
        btn_af = QtWidgets.QPushButton("Browse…")
        def _pick_adapt_folder():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select adaptation folder", ROOT_DIR)
            if d: self.adapt_folder.setText(d)
        btn_af.clicked.connect(_pick_adapt_folder)
        rowf = QtWidgets.QHBoxLayout(); rowf.addWidget(self.adapt_folder); rowf.addWidget(btn_af)
        form.addRow("Adaptation folder (.wav+.txt pairs):", rowf)

        # Freezes
        self.ad_freeze_enc  = QtWidgets.QCheckBox("Freeze encoder")
        self.ad_freeze_post = QtWidgets.QCheckBox("Freeze postnet")
        form.addRow("", self.ad_freeze_enc)
        form.addRow("", self.ad_freeze_post)

        # Hyperparameters
        self.ad_lr_mult    = QtWidgets.QDoubleSpinBox(); self.ad_lr_mult.setRange(0.01, 10.0); self.ad_lr_mult.setSingleStep(0.05); self.ad_lr_mult.setValue(1.0)
        self.ad_layers     = QtWidgets.QLineEdit("")  # comma list (e.g., embedding,encoder)
        self.adapt_epochs  = QtWidgets.QSpinBox(); self.adapt_epochs.setRange(1, 10000); self.adapt_epochs.setValue(self.state.get("adapt_epochs", 5))
        self.adapt_lr      = QtWidgets.QDoubleSpinBox(); self.adapt_lr.setDecimals(6); self.adapt_lr.setRange(1e-6, 1.0); self.adapt_lr.setSingleStep(1e-4); self.adapt_lr.setValue(self.state.get("adapt_lr", 1e-4))

        form.addRow("LR multiplier:", self.ad_lr_mult)
        form.addRow("Transfer layers (comma-sep):", self.ad_layers)
        form.addRow("Epochs:", self.adapt_epochs)
        form.addRow("Learning rate:", self.adapt_lr)

        # Tooltips
        self.ad_warm.setToolTip("Teacher Tacotron2 checkpoint to warm-start from (.pt file).")
        self.adapt_folder.setToolTip("Folder containing pairs of .wav and .txt with the same basename.")
        self.ad_freeze_enc.setToolTip("Freeze encoder layers during fine-tune to preserve text alignment.")
        self.ad_freeze_post.setToolTip("Freeze postnet (mel refinement) to stabilize fine-tuning.")
        self.ad_lr_mult.setToolTip("Learning-rate multiplier. <1.0 (e.g., 0.5) for gentle adaptation.")
        self.adapt_epochs.setToolTip("Epochs for adaptation run.")
        self.adapt_lr.setToolTip("Base learning rate during adaptation.")
        self.ad_layers.setToolTip("Comma list of layer prefixes to transfer (e.g. embedding,encoder).")

        v.addLayout(form)

        # Buttons row (Load/Apply/Help)
        btns = QtWidgets.QHBoxLayout()
        self.ad_load_btn  = QtWidgets.QPushButton("Load from config.json")
        self.ad_apply_btn = QtWidgets.QPushButton("Apply && Save")
        self.ad_help_btn  = QtWidgets.QPushButton("Help")
        btns.addWidget(self.ad_load_btn)
        btns.addWidget(self.ad_apply_btn)
        btns.addStretch(1)
        btns.addWidget(self.ad_help_btn)     # ensure it's actually added
        v.addLayout(btns)

        self.ad_status = QtWidgets.QLabel("")
        v.addWidget(self.ad_status)
        
        run_row = QtWidgets.QHBoxLayout()
        self.ad_start_btn = QtWidgets.QPushButton("Start Adaptation")
        self.ad_stop_btn  = QtWidgets.QPushButton("Stop")
        self.ad_stop_btn.setEnabled(False)
        run_row.addWidget(self.ad_start_btn)
        run_row.addWidget(self.ad_stop_btn)
        run_row.addStretch(1)
        v.addLayout(run_row)

        self.ad_start_btn.clicked.connect(self._on_start_adaptation)
        self.ad_stop_btn.clicked.connect(self._on_stop_adaptation)

        # Wiring
        self.ad_load_btn.clicked.connect(self._adaptation_load_from_cfg)
        self.ad_apply_btn.clicked.connect(self._adaptation_apply_save)
        self.ad_help_btn.clicked.connect(lambda: self._show_help("adaptation"))

        return w



    def _build_transcripts_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.transcripts_view = QtWidgets.QPlainTextEdit()
        self.transcripts_view.setReadOnly(True)
        v.addWidget(self.transcripts_view)
        self._refresh_transcripts()
        return w


    # ------------------------------ Actions ------------------------------

    def _print_device_banner(self):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            msg = f"CUDA available: True | GPU count: {torch.cuda.device_count()} | Device 0: {name}"
        else:
            msg = "CUDA available: False | Using CPU"
        self.status_label.setText(msg)

    def _pick_file(self, line_edit: QtWidgets.QLineEdit):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose file", "", "All Files (*)")
        if path:
            line_edit.setText(path)

    def _pick_dir(self, line_edit: QtWidgets.QLineEdit):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder", "")
        if path:
            line_edit.setText(path)

    # --- Synthesis
    def _on_synthesize(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            self._info("Enter some text.")
            return
        taco = self.taco_path.text().strip()
        wg = self.wg_path.text().strip()
        device = self.device_combo.currentText()
        sigma = self.sigma_slider.value() / 100.0
        denoise = self.denoise_slider.value() / 100.0

        self._save_ui_state()

        self.progress.setValue(0)
        self.status_label.setText("Starting synthesis…")
        self.synth_worker = SynthWorker(text, taco, wg, device, denoise, sigma, SYNTH_OUT_WAV)
        self.synth_worker.progress.connect(self._on_progress)
        self.synth_worker.finished.connect(self._on_synth_done)
        self.synth_worker.error.connect(self._on_error)
        self.synth_worker.start()

    def _on_progress(self, pct, msg):
        self.progress.setValue(int(pct))
        if msg:
            self.status_label.setText(msg)

    def _on_synth_done(self, path):
        self.status_label.setText(f"Done: {path}")

    def _on_error(self, msg):
        self._error(msg)

    def _on_play(self):
        if not os.path.exists(SYNTH_OUT_WAV):
            self._info("No audio synthesized yet.")
            return
        url = QtCore.QUrl.fromLocalFile(os.path.abspath(SYNTH_OUT_WAV))
        self.player.setMedia(QtMultimedia.QMediaContent(url))
        self.player.play()
        self.status_label.setText("Playing audio…")

    def _on_save_as(self):
        if not os.path.exists(SYNTH_OUT_WAV):
            self._info("No audio synthesized yet.")
            return
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save WAV", "output.wav", "WAV files (*.wav)")
        if out:
            shutil.copyfile(SYNTH_OUT_WAV, out)
            self.status_label.setText(f"Saved: {out}")

    # --- Dataset
    def on_dataset_generate(self):
        ds_root = self.ds_dir_edit.text().strip()
        if not ds_root:
            QtWidgets.QMessageBox.warning(self, "Select dataset", "Pick a dataset folder (must contain 'wavs/').")
            return
        out_filelists = os.path.join(ROOT_DIR, "filelists")

        self.ds_status.setText("Working…")
        self.ds_run_btn.setEnabled(False)

        self.ds_thread = QtCore.QThread()
        self.ds_worker = DatasetWorker(
            dataset_dir=ds_root,
            out_filelists=out_filelists,
            resample_22050=self.ds_resample.isChecked(),
            val_ratio=0.02,
            whisper_model="small"
        )
        self.ds_worker.moveToThread(self.ds_thread)
        self.ds_thread.started.connect(self.ds_worker.run)
        self.ds_worker.progress.connect(self.ds_status.setText)
        self.ds_worker.finished.connect(self._ds_done)
        self.ds_worker.failed.connect(self._ds_fail)
        self.ds_worker.finished.connect(self.ds_thread.quit)
        self.ds_worker.failed.connect(self.ds_thread.quit)
        self.ds_worker.finished.connect(self.ds_worker.deleteLater)
        self.ds_worker.failed.connect(self.ds_worker.deleteLater)
        self.ds_thread.finished.connect(self.ds_thread.deleteLater)
        self.ds_thread.start()

    def _ds_done(self, msg: str):
        self.ds_status.setText(msg)
        self.ds_run_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Dataset", msg)

    def _ds_fail(self, msg: str):
        logger.error("Dataset worker failed: %s", msg)
        self.ds_status.setText("Failed.")
        self.ds_run_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Dataset error", msg)



    def _start_record(self):
        if not PYAUDIO_OK:
            self._error("PyAudio not available.")
            return
        _, wavs_dir, _ = dataset_paths()
        self.rec = RecorderThread(wavs_dir, sr=44100)
        self.rec.recording_started.connect(lambda: self.status_label.setText("Recording…"))
        self.rec.progress.connect(self._on_progress)
        self.rec.error.connect(self._on_error)
        self.rec.recording_stopped.connect(self._on_recording_stopped)
        self.rec.start()

    def _stop_record(self):
        if hasattr(self, "rec") and self.rec.isRunning():
            self.rec.stop()

    # --- replace the whole function with this version ---
    def _on_recording_stopped(self, big_wav_path: str):
        """
        Post-process a long take by:
          1) Peak-normalizing
          2) Truncating long silences (Audacity Truncate Silence emulation)
             - Threshold:  -20 dB
             - Minimum:     0.5 s
             - Truncate to: 0.5 s
          3) Splitting into 8–12-sentence clips (uses Whisper timestamps if available)
          4) Saving as LJSpeech-style filenames and appending rows to metadata.csv
        """
        import subprocess, sys, os
        from pathlib import Path

        self.status_label.setText("Processing recording…")
        self.progress.setValue(0)

        try:
            # Resolve dataset paths and helper script location
            ds_root, wavs_dir, meta_csv = dataset_paths()
            script_path = os.path.join(os.path.dirname(__file__), "postprocess_recording.py")
            if not os.path.exists(script_path):
                raise FileNotFoundError(
                    f"postprocess_recording.py not found at: {script_path}\n"
                    "Place it in the same folder as this app."
                )

            # Prefer Whisper if available (the helper script will still work without it)
            use_whisper_flag = "--use_whisper" if WHISPER_OK else ""

            # Run the helper script (normalizes, truncates silence, makes 8–12-sentence chunks,
            # writes LJ*-*.wav files to wavs/ and appends to metadata.csv)
            cmd = [
                sys.executable, script_path,
                "--input_wav", big_wav_path,
                "--dataset_root", ds_root,
            ]
            if use_whisper_flag:
                cmd.append(use_whisper_flag)

            self.status_label.setText("Truncating silence and splitting by sentences…")
            # Show progress bar pulse during external processing
            self.progress.setRange(0, 0)  # indeterminate
            subprocess.check_call(cmd)

            # Cleanup the big raw file (best-effort)
            try:
                os.remove(big_wav_path)
            except Exception:
                pass

            # Refresh UI and finish
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self._refresh_audio_list()
            self._refresh_transcripts()
            self.status_label.setText("Recording processed (silence-truncated and sentence-split).")

        except subprocess.CalledProcessError as e:
            self.progress.setRange(0, 100)
            self._error(f"Post-processing script failed with exit code {e.returncode}.")
        except Exception as e:
            self.progress.setRange(0, 100)
            self._error(f"Post-processing failed: {e}")


    def _upload_audio_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose audio", "", "Audio (*.wav *.flac *.mp3);;All (*)")
        if not paths:
            return
        _, wavs_dir, _ = dataset_paths()
        for p in paths:
            dst = os.path.join(wavs_dir, os.path.basename(p))
            shutil.copyfile(p, dst)
        self._refresh_audio_list()
        self.status_label.setText(f"Copied {len(paths)} file(s).")

    def _upload_transcript(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose transcript", "", "Text (*.txt);;All (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                self._info("Transcript file is empty.")
                return
            # Append whole file as one entry (or split by lines if you prefer)
            if tts_model and hasattr(tts_model, "update_metadata_file"):
                # requires a wav name; here we insert a placeholder if needed
                wav_name = f"uploaded_{int(time.time())}.wav"
                tts_model.update_metadata_file(wav_name, content)
            self._refresh_transcripts()
            self.status_label.setText("Transcript appended to metadata.csv.")
        except Exception as e:
            self._error(f"Failed to append transcript: {e}")

    def _refresh_audio_list(self):
        _, wavs_dir, _ = dataset_paths()
        self.list_audio.clear()
        if os.path.isdir(wavs_dir):
            for fn in sorted(os.listdir(wavs_dir)):
                if fn.lower().endswith((".wav", ".flac", ".mp3")):
                    self.list_audio.addItem(fn)

    # --- Training
    def _start_training(self):
        if tts_model is None:
            self._error("tts_model not available.")
            return
        out_dir = self.train_out_dir.text().strip() or os.path.join("datasets", "output")
        # Build config object
        ds_root, wavs_dir, meta_csv = dataset_paths()
        cfg = {
            "datasets": [{"path": ds_root, "wavs": wavs_dir, "metadata": meta_csv}],
            "training": {
                "batch_size": int(self.tr_bs.value()),
                "num_workers": 0 if sys.platform.startswith("win") else int(self.opt_workers.value()),
                "learning_rate": float(self.opt_lr.value())
            }
        }
        self.progress.setValue(0)
        self.status_label.setText("Starting training…")
        self.train_thread = TrainingThread(cfg, out_dir)
        self.train_thread.progress.connect(self._on_progress)
        self.train_thread.finished.connect(lambda d: self.status_label.setText(f"Training finished: {d}"))
        self.train_thread.error.connect(self._on_error)
        self.train_thread.start()

    # --- Adaptation
    def _run_adaptation(self):
        if tts_model is None:
            self._error("tts_model not available.")
            return
        folder = self.adapt_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self._error("Choose a valid adaptation folder.")
            return

        # Scan pairs
        pairs: List[Tuple[str, torch.Tensor]] = []
        try:
            for fn in os.listdir(folder):
                if not fn.lower().endswith(".wav"):
                    continue
                base = fn[:-4]
                txt = os.path.join(folder, base + ".txt")
                if not os.path.exists(txt):
                    continue
                with open(txt, "r", encoding="utf-8") as f:
                    trans = f.read().strip()
                wav_path = os.path.join(folder, fn)
                wav = tts_model.load_wav(wav_path, sr=APP_SR)
                mel = tts_model.compute_mel_spectrogram(wav, sr=APP_SR)
                pairs.append((trans, mel))
        except Exception as e:
            self._error(f"Failed collecting adaptation data: {e}")
            return

        if not pairs:
            self._info("No .wav+.txt pairs found.")
            return

        self.status_label.setText(f"Adapting on {len(pairs)} pairs…")
        try:
            # Use current Tacotron as base if possible
            base_model = None
            if hasattr(self, "synth_worker"):
                pass
            # user-code: derive char_to_idx from NvSynth or your tts_model vocab
            char_to_idx = {c: i+1 for i, c in enumerate(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;?&")}
            if hasattr(tts_model, "adapt_speaker"):
                # Train briefly and replace tacotron checkpoint on disk (optional)
                model = tts_model.Tacotron2(vocab_size=len(char_to_idx)+1, mel_channels=80, max_len=1200)
                model = tts_model.adapt_speaker(model, pairs, char_to_idx=char_to_idx,
                                                num_epochs=int(self.adapt_epochs.value()),
                                                lr=float(self.adapt_lr.value()))
                self.status_label.setText("Adaptation finished and model updated.")
            else:
                self._info("tts_model.adapt_speaker not implemented; skipped.")
        except Exception as e:
            self._error(f"Adaptation failed: {e}")

    # --- Transcripts
    def _refresh_transcripts(self):
        _, _, meta_csv = dataset_paths()
        try:
            if os.path.exists(meta_csv):
                with open(meta_csv, "r", encoding="utf-8") as f:
                    self.transcripts_view.setPlainText(f.read())
            else:
                self.transcripts_view.setPlainText("(metadata.csv not found)")
        except Exception as e:
            self.transcripts_view.setPlainText(f"(Failed to load metadata: {e})")


    # ------------------------------ Helpers ------------------------------

    def _save_ui_state(self):
        def val(name, default=None, attr=None, getter="text"):
            obj = getattr(self, attr or name, None)
            if obj is None:
                return default
            if hasattr(obj, getter):
                try:
                    v = getattr(obj, getter)()
                    return v.strip() if isinstance(v, str) else v
                except Exception:
                    return default
            return default

        state = {
            # Synthesis
            "taco_path": val("taco_path", ""),
            "wg_path":   val("wg_path", ""),
            "device":    self.device_combo.currentText() if hasattr(self, "device_combo") else "cpu",
            "sigma":     self.sigma_slider.value() if hasattr(self, "sigma_slider") else 60,
            "denoise":   self.denoise_slider.value() if hasattr(self, "denoise_slider") else 0,
            "last_text": self.text_edit.toPlainText() if hasattr(self, "text_edit") else "",
            # Training (only if present)
            "train_out_dir": val("train_out_dir", os.path.join("checkpoints", "runs", "run1")),
            "tr_bs": int(self.tr_bs.value()) if hasattr(self, "tr_bs") else 32,
            "opt_workers": int(self.opt_workers.value()) if hasattr(self, "opt_workers") else (0 if sys.platform.startswith("win") else 4),
            "opt_lr": float(self.opt_lr.value()) if hasattr(self, "opt_lr") else 1e-3,
            "tr_cudnn_bench": bool(self.tr_cudnn_bench.isChecked()) if hasattr(self, "tr_cudnn_bench") else False,
            # Adaptation (only if present)
            "adapt_folder": val("adapt_folder", ""),
            "adapt_epochs": int(self.adapt_epochs.value()) if hasattr(self, "adapt_epochs") else 5,
            "adapt_lr": float(self.adapt_lr.value()) if hasattr(self, "adapt_lr") else 1e-4,
        }
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        _write_json(self.settings_path, state)


    def _restore_ui_state(self):
        # already loaded into self.state during __init__
        pass

    def _info(self, msg: str):
        QtWidgets.QMessageBox.information(self, "Info", msg)
        self.status_label.setText(msg)

    def _error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self.status_label.setText(msg)

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
