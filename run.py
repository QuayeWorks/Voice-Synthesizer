"""PyQt5 GUI for QWPitch (text->mel) + QWGAN (mel->wav).

This GUI keeps the modern pipeline only (no Tacotron2/WaveGlow).
- QWPitch produces mels directly from text.
- QWGAN vocodes those mels into WAV files.

Features
--------
- Model selection for QWPitch checkpoint, QWGAN generator + config.
- CUDA + AMP toggles (AMP only affects QWPitch when CUDA is on).
- Pace + pitch transform controls for QWPitch.
- Multi-line text input (one line = one output wav) with batch processing.
- Optional mel saving alongside WAVs.
- Output directory picker with /mels + /wavs subfolders.
- Progress log panel with per-step updates.
- Global naming to avoid overwritten outputs.
"""
from __future__ import annotations

import sys
import os

# ------------------------------------------------------------------
# CRITICAL FIX:
# QWPitch parses CLI args at *import time*.
# When running a GUI, we must strip argv BEFORE importing QWPitch.
# ------------------------------------------------------------------
sys.argv = [sys.argv[0]]

import argparse
import datetime as dt
from dataclasses import dataclass
from contextlib import nullcontext
import shlex
import signal
import subprocess
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from fastpitch_config import QWPitchProjectConfig, TokenizerConfig, load_qwpitch_config

import numpy as np
import torch
from scipy.io.wavfile import write


def _configure_qt_platform():
    """Pick a sensible Qt platform plugin based on the environment.

    Goal: avoid the common "Could not load the Qt platform plugin xcb" crash,
    especially on WSL/WSLg and minimal installs.

    Rules:
    - If QT_QPA_PLATFORM is explicitly set, respect it.
    - If WAYLAND_DISPLAY is set (WSLg/Wayland), prefer the Wayland backend.
    - Else if DISPLAY is set (X11), let Qt choose its default (typically xcb).
    - Else (headless), force the offscreen backend.
    """

    if os.getenv("QT_QPA_PLATFORM"):
        return

    if os.getenv("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        return

    if os.getenv("DISPLAY"):
        return

    os.environ["QT_QPA_PLATFORM"] = "offscreen"


_configure_qt_platform()

from PyQt5 import QtCore, QtWidgets, QtGui  # noqa: E402

# QWPitch + QWGAN imports
from third_party.fastpitch import inference as fp_infer  # noqa: E402
from third_party.fastpitch.common.text import cmudict  # noqa: E402
from third_party.hifigan.env import AttrDict  # noqa: E402
from third_party.hifigan.models import Generator  # noqa: E402

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_CONFIG: QWPitchProjectConfig = load_qwpitch_config()


def summarize_tokenizer(separator: str = " | ") -> str:
    return separator.join(PROJECT_CONFIG.tokenizer.summary_lines())


@dataclass
class PitchSettings:
    flatten: bool = False
    invert: bool = False
    amplify: float = 1.0
    shift_hz: float = 0.0
    custom: bool = False


@dataclass
class QWPitchOutput:
    stem: str
    mel: np.ndarray  # (T, n_mels)
    mel_path: Optional[Path]


# ------------------------- Inference helpers ------------------------- #


def _build_pitch_transform(settings: PitchSettings):
    dummy_parser = argparse.Namespace(
        pitch_transform_flatten=settings.flatten,
        pitch_transform_invert=settings.invert,
        pitch_transform_amplify=settings.amplify,
        pitch_transform_shift=settings.shift_hz,
        pitch_transform_custom=settings.custom,
    )
    return fp_infer.build_pitch_transformation(dummy_parser)


def _prepare_fastpitch_model(ckpt: Path, device: torch.device, use_amp: bool):
    """Load QWPitch without requiring CLI args.

    NOTE: third_party.fastpitch.inference.parse_args() defines required CLI flags
    (e.g., -i/--input and --fastpitch). When we call argparse with an empty argv
    inside the GUI, argparse exits with a usage error.

    To keep using QWPitch's helper utilities (EMA/torchscript flags, etc.)
    while running from the GUI, we pass a minimal dummy argv that satisfies
    those required arguments. The GUI still controls the actual checkpoint
    path via the `ckpt` parameter passed into load_and_setup_model().
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser = fp_infer.parse_args(parser)

    # Provide minimal required CLI args so argparse doesn't error out when running
    # from the GUI. These values are not used for synthesis in the GUI path.
    dummy_argv = ["-i", "__gui_input_unused__", "--fastpitch", str(ckpt)]
    args, unk_args = parser.parse_known_args(dummy_argv)

    args.amp = use_amp
    model = fp_infer.load_and_setup_model(
        "QWPitch",
        parser,
        str(ckpt),
        use_amp,
        device,
        unk_args=unk_args,
        forward_is_infer=True,
        ema=getattr(args, "ema", False),
        jitable=getattr(args, "torchscript", False),
    )
    return model


def fastpitch_infer(
    texts: Sequence[str],
    stems: Sequence[str],
    checkpoint: Path,
    device: torch.device,
    pace: float,
    pitch: PitchSettings,
    save_mels: bool,
    mel_dir: Path,
    batch_size: int = 8,
    use_amp: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    tokenizer_cfg: TokenizerConfig | None = None,
) -> List[QWPitchOutput]:
    """Run QWPitch on provided text lines and return mel outputs."""

    if len(texts) != len(stems):
        raise ValueError("texts and stems must have the same length")

    if progress:
        progress("Loading QWPitch…")

    if any(len(t.strip()) == 0 for t in texts):
        raise ValueError("Blank lines are not allowed in the input text")

    amp_enabled = use_amp and device.type == "cuda"
    model = _prepare_fastpitch_model(checkpoint, device, amp_enabled)

    if progress:
        progress("Preparing input batches…")

    tokenizer_cfg = tokenizer_cfg or PROJECT_CONFIG.tokenizer
    fields = {"text": list(texts), "output": list(stems)}
    batches = fp_infer.prepare_input_sequence(
        fields,
        device,
        symbol_set=tokenizer_cfg.symbol_set,
        text_cleaners=tokenizer_cfg.text_cleaners,
        batch_size=batch_size,
        dataset=None,
        load_mels=False,
        p_arpabet=tokenizer_cfg.p_arpabet,
        include_style_tokens=tokenizer_cfg.include_style_tokens,
        style_tags=tokenizer_cfg.style_tags,
        strip_style_from_text=tokenizer_cfg.strip_style_from_text,
    )

    pitch_transform = _build_pitch_transform(pitch)

    gen_kw = {
        "pace": pace,
        "speaker": 0,
        "pitch_tgt": None,
        "pitch_transform": pitch_transform,
    }

    outputs: List[QWPitchOutput] = []
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext

    for batch in batches:
        with torch.no_grad():
            with autocast(enabled=amp_enabled) if device.type == "cuda" else autocast():
                mel, mel_lens, *_ = model(batch["text"], **gen_kw)

        for i, mel_tensor in enumerate(mel):
            mel_trimmed = mel_tensor[:, : mel_lens[i].item()].permute(1, 0).cpu().numpy()
            stem = batch["output"][i]
            mel_path = None
            if save_mels:
                mel_dir.mkdir(parents=True, exist_ok=True)
                mel_path = mel_dir / f"{stem}.npy"
                np.save(mel_path, mel_trimmed)
            outputs.append(QWPitchOutput(stem=stem, mel=mel_trimmed, mel_path=mel_path))

        if progress:
            progress(f"Processed batch with {len(batch['text'])} line(s)")

    return outputs


def _load_hifigan(generator_path: Path, config_path: Path, device: torch.device):
    import json

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    hparams = AttrDict(cfg)
    generator = Generator(hparams).to(device)

    state = torch.load(generator_path, map_location=device)
    if "generator" in state:
        generator.load_state_dict(state["generator"])
    elif "state_dict" in state:
        generator.load_state_dict(state["state_dict"], strict=False)
    else:
        generator.load_state_dict(state, strict=False)

    generator.eval()
    generator.remove_weight_norm()
    return generator, hparams


def hifigan_vocode(
    mels: Iterable[QWPitchOutput],
    generator_path: Path,
    config_path: Path,
    out_dir: Path,
    device: torch.device,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Path]:
    """Convert mels to wavs using QWGAN."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Loading QWGAN…")

    generator, hparams = _load_hifigan(generator_path, config_path, device)
    wav_paths: List[Path] = []

    with torch.no_grad():
        for item in mels:
            mel = torch.from_numpy(item.mel.T).unsqueeze(0).to(device).float()
            audio = generator(mel).squeeze().cpu().numpy()
            audio = audio / max(1e-8, np.max(np.abs(audio)))
            wav16 = (audio * 32767.0).astype(np.int16)

            out_path = out_dir / f"{item.stem}.wav"
            write(str(out_path), int(hparams.sampling_rate), wav16)
            wav_paths.append(out_path)

            if progress:
                progress(f"Saved {out_path.relative_to(BASE_DIR)}")

    return wav_paths


def run_pipeline(
    text_lines: Sequence[str],
    fastpitch_ckpt: Path,
    hifigan_ckpt: Path,
    hifigan_cfg: Path,
    output_root: Path,
    use_cuda: bool,
    use_amp: bool,
    pace: float,
    pitch: PitchSettings,
    save_mels: bool,
    batch_size: int,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Path]:
    """Full text -> mel -> wav pipeline."""

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    if use_cuda and device.type != "cuda" and progress:
        progress("CUDA requested but not available; falling back to CPU")

    if progress:
        progress(f"Running on {device}")
        progress(f"Tokenizer: {summarize_tokenizer()}")

    output_root.mkdir(parents=True, exist_ok=True)
    mel_dir = output_root / "mels"
    wav_dir = output_root / "wavs"

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_counter = len(list(wav_dir.glob("*.wav")))
    stems = [f"{timestamp}_{base_counter + i:05d}" for i in range(len(text_lines))]

    fp_outputs = fastpitch_infer(
        texts=text_lines,
        stems=stems,
        checkpoint=fastpitch_ckpt,
        device=device,
        pace=pace,
        pitch=pitch,
        save_mels=save_mels,
        mel_dir=mel_dir,
        batch_size=batch_size,
        use_amp=use_amp,
        progress=progress,
    )

    if progress:
        progress(f"QWPitch finished. Vocoding {len(fp_outputs)} mel(s)…")

    wav_paths = hifigan_vocode(
        fp_outputs,
        generator_path=hifigan_ckpt,
        config_path=hifigan_cfg,
        out_dir=wav_dir,
        device=device,
        progress=progress,
    )

    return wav_paths


# ------------------------------- GUI -------------------------------- #


class ProcessWorker(QtCore.QThread):
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(int)

    def __init__(self, cmd: list[str]):
        super().__init__()
        self.cmd = cmd
        self._proc: Optional[subprocess.Popen[str]] = None

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.terminate()
                except Exception:  # noqa: BLE001
                    pass

    def run(self):  # noqa: D401
        """Spawn the subprocess and stream its output."""

        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_line.emit(f"Failed to start process: {exc}")
            self.finished.emit(1)
            return

        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self.log_line.emit(line.rstrip("\n"))

        self._proc.wait()
        self.finished.emit(self._proc.returncode or 0)


class SynthWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(
        self,
        text: str,
        fastpitch_ckpt: Path,
        hifigan_ckpt: Path,
        hifigan_config: Path,
        output_dir: Path,
        use_cuda: bool,
        use_amp: bool,
        pace: float,
        pitch: PitchSettings,
        save_mels: bool,
        batch_size: int,
    ):
        super().__init__()
        self.text = text
        self.fastpitch_ckpt = Path(fastpitch_ckpt).resolve()
        self.hifigan_ckpt = Path(hifigan_ckpt).resolve()
        self.hifigan_config = Path(hifigan_config).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.use_cuda = use_cuda
        self.use_amp = use_amp
        self.pace = pace
        self.pitch = pitch
        self.save_mels = save_mels
        self.batch_size = batch_size

    def run(self):
        try:
            lines = [ln.strip() for ln in self.text.splitlines() if ln.strip()]
            if not lines:
                raise ValueError("Please enter at least one non-empty line of text.")

            cmudict.initialize(str(BASE_DIR / "third_party/fastpitch/cmudict/cmudict-0.7b"), keep_ambiguous=True)

            wav_paths = run_pipeline(
                text_lines=lines,
                fastpitch_ckpt=self.fastpitch_ckpt,
                hifigan_ckpt=self.hifigan_ckpt,
                hifigan_cfg=self.hifigan_config,
                output_root=self.output_dir,
                use_cuda=self.use_cuda,
                use_amp=self.use_amp,
                pace=self.pace,
                pitch=self.pitch,
                save_mels=self.save_mels,
                batch_size=self.batch_size,
                progress=self.progress.emit,
            )

            summary = "\n".join(str(p.relative_to(BASE_DIR)) for p in wav_paths)
            self.finished.emit(True, f"Generated {len(wav_paths)} file(s):\n{summary}")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QWPitch + QWGAN GUI")
        self.worker: Optional[SynthWorker] = None
        self.process_worker: Optional[ProcessWorker] = None
        self.settings = QtCore.QSettings("VoiceSynthesizer", "QWPitchHiFiGUI")

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_synthesis_tab(), "Synthesis")
        self.tabs.addTab(self._build_training_tab(), "Training")

        outer = QtWidgets.QVBoxLayout()
        outer.addWidget(self.tabs)
        self.setLayout(outer)

        self._load_settings()

    def _build_synthesis_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()

        # Inputs
        self.fastpitch_edit = QtWidgets.QLineEdit()
        self.hifigan_edit = QtWidgets.QLineEdit()
        self.hifigan_cfg_edit = QtWidgets.QLineEdit()

        # FIX: create this before using it
        self.input_file_edit = QtWidgets.QLineEdit()
        self.input_file_edit.setPlaceholderText("Path to phrases .txt file (one line per phrase)")

        default_out = (BASE_DIR / "inference").resolve()
        self.output_edit = QtWidgets.QLineEdit(str(default_out))

        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setPlaceholderText("Enter one phrase per line")

        self.tokenizer_label = QtWidgets.QLabel(summarize_tokenizer("\n"))
        self.tokenizer_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.tokenizer_label.setWordWrap(True)

        self.cuda_check = QtWidgets.QCheckBox("Use CUDA if available")
        self.cuda_check.setChecked(torch.cuda.is_available())
        self.amp_check = QtWidgets.QCheckBox("Use AMP (QWPitch)")
        self.amp_check.setToolTip("Enabled only when CUDA is available")

        self.pace_spin = QtWidgets.QDoubleSpinBox()
        self.pace_spin.setRange(0.1, 3.0)
        self.pace_spin.setSingleStep(0.05)
        self.pace_spin.setValue(1.0)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(4)

        self.save_mels_check = QtWidgets.QCheckBox("Save mel .npy files")
        self.pitch_flat_check = QtWidgets.QCheckBox("Flatten pitch")
        self.pitch_invert_check = QtWidgets.QCheckBox("Invert pitch")
        self.pitch_custom_check = QtWidgets.QCheckBox("Custom transform (pitch_transform.py)")
        self.pitch_amplify_spin = QtWidgets.QDoubleSpinBox()
        self.pitch_amplify_spin.setRange(0.1, 5.0)
        self.pitch_amplify_spin.setSingleStep(0.1)
        self.pitch_amplify_spin.setValue(1.0)
        self.pitch_shift_spin = QtWidgets.QDoubleSpinBox()
        self.pitch_shift_spin.setRange(-400.0, 400.0)
        self.pitch_shift_spin.setSingleStep(5.0)
        self.pitch_shift_spin.setValue(0.0)
        self.pitch_shift_spin.setSuffix(" Hz")

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)

        synth_btn = QtWidgets.QPushButton("Synthesize")
        synth_btn.clicked.connect(self.start_synthesis)

        layout = QtWidgets.QFormLayout()
        for label, widget, handler in (
            ("QWPitch checkpoint", self.fastpitch_edit, self.browse_model),
            ("QWGAN checkpoint", self.hifigan_edit, self.browse_hifigan),
            ("QWGAN config", self.hifigan_cfg_edit, self.browse_hifigan_cfg),
            ("Phrases file (-i)", self.input_file_edit, self.browse_input_file),
            ("Output directory", self.output_edit, self.browse_output),
        ):
            layout.addRow(label, self._row(widget, handler))

        layout.addRow("Text", self.text_edit)
        layout.addRow("Tokenizer", self.tokenizer_label)
        layout.addRow(self.cuda_check, self.amp_check)
        layout.addRow("Pace", self.pace_spin)
        layout.addRow("Batch size", self.batch_spin)
        layout.addRow(self.save_mels_check)

        pitch_box = QtWidgets.QVBoxLayout()
        pitch_box.addWidget(self.pitch_flat_check)
        pitch_box.addWidget(self.pitch_invert_check)
        pitch_box.addWidget(self.pitch_custom_check)
        pitch_box.addWidget(self._labeled_row("Amplify", self.pitch_amplify_spin))
        pitch_box.addWidget(self._labeled_row("Shift", self.pitch_shift_spin))
        pitch_widget = QtWidgets.QWidget()
        pitch_widget.setLayout(pitch_box)
        layout.addRow("Pitch transforms", pitch_widget)

        layout.addRow(synth_btn)
        layout.addRow("Log", self.log_box)

        content.setLayout(layout)
        scroll.setWidget(content)

        outer = QtWidgets.QVBoxLayout()
        outer.addWidget(scroll)
        tab.setLayout(outer)
        return tab

    def _build_training_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()

        warning = QtWidgets.QLabel(
            "If resuming from a checkpoint, ensure n_symbols matches your current tokenizer (158).\n"
            "If vocab differs, retrain from scratch or provide compatible symbols."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #b58900; font-weight: bold;")
        vbox.addWidget(warning)

        # Paths & files
        paths_group = QtWidgets.QGroupBox("Paths & files")
        paths_form = QtWidgets.QFormLayout()
        self.dataset_root_edit = QtWidgets.QLineEdit(str(BASE_DIR / "datasets/LJSpeech-1.1"))
        self.train_filelist_edit = QtWidgets.QLineEdit(str(BASE_DIR / "filelists/ljs_train_clean.txt"))
        self.val_filelist_edit = QtWidgets.QLineEdit(str(BASE_DIR / "filelists/ljs_val_clean.txt"))
        self.ckpt_dir_edit = QtWidgets.QLineEdit(str(BASE_DIR / "checkpoints/fastpitch_ddp_test"))
        self.fastpitch_folder_edit = QtWidgets.QLineEdit(str(BASE_DIR / "third_party/fastpitch"))
        self.python_exec_edit = QtWidgets.QLineEdit(sys.executable)
        self.use_venv_python_check = QtWidgets.QCheckBox("Use current venv python")
        self.use_venv_python_check.setChecked(True)
        self.torchrun_exec_edit = QtWidgets.QLineEdit("torchrun")

        def add_path_row(label: str, widget: QtWidgets.QLineEdit, directory: bool = True):
            btn = QtWidgets.QPushButton("Browse")
            if directory:
                btn.clicked.connect(lambda _, w=widget: self._pick_dir(w))
            else:
                btn.clicked.connect(lambda _, w=widget: self._pick_file(w))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(widget)
            row.addWidget(btn)
            container = QtWidgets.QWidget()
            container.setLayout(row)
            paths_form.addRow(label, container)

        add_path_row("Dataset root (-d)", self.dataset_root_edit)
        add_path_row("Train filelist", self.train_filelist_edit, directory=False)
        add_path_row("Val filelist", self.val_filelist_edit, directory=False)
        add_path_row("Output checkpoints", self.ckpt_dir_edit)
        add_path_row("FastPitch folder", self.fastpitch_folder_edit)
        paths_form.addRow("Python executable", self.python_exec_edit)
        paths_form.addRow(self.use_venv_python_check)
        paths_form.addRow("Torchrun executable", self.torchrun_exec_edit)
        paths_group.setLayout(paths_form)
        vbox.addWidget(paths_group)

        # Precompile settings
        prep_group = QtWidgets.QGroupBox("Precompile settings")
        prep_form = QtWidgets.QFormLayout()
        self.extract_mels_check = QtWidgets.QCheckBox("Extract mels")
        self.extract_mels_check.setChecked(True)
        self.extract_pitch_check = QtWidgets.QCheckBox("Extract pitch")
        self.extract_pitch_check.setChecked(True)
        self.prep_batch_spin = QtWidgets.QSpinBox()
        self.prep_batch_spin.setRange(1, 256)
        self.prep_batch_spin.setValue(8)
        self.prep_workers_spin = QtWidgets.QSpinBox()
        self.prep_workers_spin.setRange(1, 128)
        self.prep_workers_spin.setValue(16)
        prep_btn = QtWidgets.QPushButton("Run Precompile")
        prep_btn.clicked.connect(self.start_precompile)
        prep_form.addRow(self.extract_mels_check, self.extract_pitch_check)
        prep_form.addRow("Batch size (-b)", self.prep_batch_spin)
        prep_form.addRow("Workers (--n-workers)", self.prep_workers_spin)
        prep_form.addRow(prep_btn)
        prep_group.setLayout(prep_form)
        vbox.addWidget(prep_group)

        # Training settings
        train_group = QtWidgets.QGroupBox("Training settings")
        train_form = QtWidgets.QFormLayout()
        self.nproc_spin = QtWidgets.QSpinBox()
        self.nproc_spin.setRange(1, 64)
        self.nproc_spin.setValue(2)
        self.master_port_spin = QtWidgets.QSpinBox()
        self.master_port_spin.setRange(1024, 65535)
        self.master_port_spin.setValue(29501)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(900)
        self.epc_spin = QtWidgets.QSpinBox()
        self.epc_spin.setRange(1, 1000)
        self.epc_spin.setValue(25)
        self.train_batch_spin = QtWidgets.QSpinBox()
        self.train_batch_spin.setRange(1, 1024)
        self.train_batch_spin.setValue(16)
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setValue(1e-3)
        self.num_workers_spin = QtWidgets.QSpinBox()
        self.num_workers_spin.setRange(1, 256)
        self.num_workers_spin.setValue(16)
        self.prefetch_spin = QtWidgets.QSpinBox()
        self.prefetch_spin.setRange(1, 16)
        self.prefetch_spin.setValue(2)
        self.load_mel_check = QtWidgets.QCheckBox("Load mel from disk")
        self.load_mel_check.setChecked(True)
        self.load_pitch_check = QtWidgets.QCheckBox("Load pitch from disk")
        self.load_pitch_check.setChecked(True)
        self.open_output_btn = QtWidgets.QPushButton("Open output folder")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        train_btn = QtWidgets.QPushButton("Start Training")
        train_btn.clicked.connect(self.start_training)
        self.stop_train_btn = QtWidgets.QPushButton("Stop")
        self.stop_train_btn.clicked.connect(self.stop_process)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(train_btn)
        btn_row.addWidget(self.stop_train_btn)
        btn_row.addWidget(self.open_output_btn)
        btn_container = QtWidgets.QWidget()
        btn_container.setLayout(btn_row)

        train_form.addRow("GPUs / processes", self.nproc_spin)
        train_form.addRow("master_port", self.master_port_spin)
        train_form.addRow("epochs", self.epochs_spin)
        train_form.addRow("epochs-per-checkpoint", self.epc_spin)
        train_form.addRow("batch size (-bs)", self.train_batch_spin)
        train_form.addRow("learning rate (-lr)", self.lr_spin)
        train_form.addRow("num-workers", self.num_workers_spin)
        train_form.addRow("prefetch-factor", self.prefetch_spin)
        train_form.addRow(self.load_mel_check, self.load_pitch_check)
        train_form.addRow(btn_container)
        train_group.setLayout(train_form)
        vbox.addWidget(train_group)

        # Text/tokenization controls
        text_group = QtWidgets.QGroupBox("Text/tokenization consistency")
        text_form = QtWidgets.QFormLayout()
        self.symbol_set_edit = QtWidgets.QLineEdit(PROJECT_CONFIG.tokenizer.symbol_set)
        self.text_cleaners_edit = QtWidgets.QLineEdit(
            ",".join(PROJECT_CONFIG.tokenizer.text_cleaners or ["english_cleaners_v2"])
        )
        self.include_style_tokens_check = QtWidgets.QCheckBox("Include style tokens")
        self.include_style_tokens_check.setChecked(PROJECT_CONFIG.tokenizer.include_style_tokens)
        self.style_tags_edit = QtWidgets.QLineEdit(
            ",".join(PROJECT_CONFIG.tokenizer.style_tags or [])
        )
        self.strip_style_check = QtWidgets.QCheckBox("Strip style from text")
        self.strip_style_check.setChecked(PROJECT_CONFIG.tokenizer.strip_style_from_text)
        text_form.addRow("--symbol-set", self.symbol_set_edit)
        text_form.addRow("--text-cleaners", self.text_cleaners_edit)
        text_form.addRow(self.include_style_tokens_check, self.strip_style_check)
        text_form.addRow("Style tags", self.style_tags_edit)
        text_group.setLayout(text_form)
        vbox.addWidget(text_group)

        # Command preview + logs
        cmd_group = QtWidgets.QGroupBox("Dry run / Show command")
        cmd_layout = QtWidgets.QVBoxLayout()
        self.command_preview = QtWidgets.QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        cmd_buttons = QtWidgets.QHBoxLayout()
        self.show_precompile_btn = QtWidgets.QPushButton("Show Precompile Cmd")
        self.show_precompile_btn.clicked.connect(lambda: self._update_command_preview(precompile=True))
        self.show_train_btn = QtWidgets.QPushButton("Show Train Cmd")
        self.show_train_btn.clicked.connect(lambda: self._update_command_preview(precompile=False))
        cmd_buttons.addWidget(self.show_precompile_btn)
        cmd_buttons.addWidget(self.show_train_btn)
        cmd_layout.addLayout(cmd_buttons)
        cmd_layout.addWidget(self.command_preview)
        cmd_group.setLayout(cmd_layout)
        vbox.addWidget(cmd_group)

        log_label = QtWidgets.QLabel("Process log")
        self.train_log = QtWidgets.QPlainTextEdit()
        self.train_log.setReadOnly(True)
        vbox.addWidget(log_label)
        vbox.addWidget(self.train_log)

        content.setLayout(vbox)
        scroll.setWidget(content)

        outer = QtWidgets.QVBoxLayout()
        outer.addWidget(scroll)
        tab.setLayout(outer)
        return tab

    def _row(self, widget: QtWidgets.QWidget, handler: Callable[[], None]):
        h = QtWidgets.QHBoxLayout()
        h.addWidget(widget)
        btn = QtWidgets.QPushButton("Browse")
        btn.clicked.connect(handler)
        h.addWidget(btn)
        row = QtWidgets.QWidget()
        row.setLayout(h)
        return row

    def _labeled_row(self, text: str, widget: QtWidgets.QWidget):
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel(text))
        h.addWidget(widget)
        container = QtWidgets.QWidget()
        container.setLayout(h)
        return container

    def browse_input_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select phrases text file",
            str(BASE_DIR),
            "Text files (*.txt)",
        )
        if path:
            self.input_file_edit.setText(path)
            self._save_settings()

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select QWPitch checkpoint", str(BASE_DIR))
        if path:
            self.fastpitch_edit.setText(path)
            self._save_settings()

    def browse_hifigan(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select QWGAN checkpoint", str(BASE_DIR))
        if path:
            self.hifigan_edit.setText(path)
            self._save_settings()

    def browse_hifigan_cfg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select QWGAN config", str(BASE_DIR))
        if path:
            self.hifigan_cfg_edit.setText(path)
            self._save_settings()

    def browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", str(BASE_DIR))
        if path:
            self.output_edit.setText(path)
            self._save_settings()

    def log(self, message: str):
        self.log_box.appendPlainText(message)

    def start_synthesis(self):
        if self.worker is not None and self.worker.isRunning():
            self.log("Synthesis already running…")
            return

        fastpitch = Path(self.fastpitch_edit.text()).resolve()
        hifigan = Path(self.hifigan_edit.text()).resolve()
        hifigan_cfg = Path(self.hifigan_cfg_edit.text()).resolve()
        output_dir = (
            Path(self.output_edit.text()).resolve()
            if self.output_edit.text()
            else (BASE_DIR / "inference").resolve()
        )

        missing = [p for p in (fastpitch, hifigan, hifigan_cfg) if not p.is_file()]
        if missing:
            self.log(f"Missing paths: {', '.join(str(m) for m in missing)}")
            return

        # Build the synthesis lines from either:
        #  - phrases file (-i), if provided
        #  - text box, otherwise
        input_file = self.input_file_edit.text().strip()
        if input_file:
            input_path = Path(input_file).expanduser().resolve()
            if not input_path.is_file():
                self.log(f"Phrases file not found: {input_path}")
                return

            lines = [
                ln.strip()
                for ln in input_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            if not lines:
                self.log("Phrases file is empty (after stripping blank lines).")
                return

            # FIX: define text for worker (worker expects a string)
            text = "\n".join(lines)
        else:
            text = self.text_edit.toPlainText()
            if not text.strip():
                self.log("Please enter text or select a phrases file (-i).")
                return

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines:
                self.log("No non-empty lines to synthesize.")
                return

        self._save_settings()
        self.log("Using paths:")
        self.log(f"  QWPitch: {fastpitch}")
        self.log(f"  QWGAN checkpoint: {hifigan}")
        self.log(f"  QWGAN config: {hifigan_cfg}")
        self.log(f"  Phrases file (-i): {input_file if input_file else '(text box)'}")
        self.log(f"  Output directory: {output_dir}")
        self.log(f"  Tokenizer: {summarize_tokenizer()}")

        pitch = PitchSettings(
            flatten=self.pitch_flat_check.isChecked(),
            invert=self.pitch_invert_check.isChecked(),
            amplify=self.pitch_amplify_spin.value(),
            shift_hz=self.pitch_shift_spin.value(),
            custom=self.pitch_custom_check.isChecked(),
        )

        self.worker = SynthWorker(
            text=text,
            fastpitch_ckpt=fastpitch,
            hifigan_ckpt=hifigan,
            hifigan_config=hifigan_cfg,
            output_dir=output_dir,
            use_cuda=self.cuda_check.isChecked(),
            use_amp=self.amp_check.isChecked(),
            pace=self.pace_spin.value(),
            pitch=pitch,
            save_mels=self.save_mels_check.isChecked(),
            batch_size=self.batch_spin.value(),
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.log("Starting synthesis…")

    def _pick_dir(self, widget: QtWidgets.QLineEdit):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory", str(BASE_DIR))
        if path:
            widget.setText(path)
            self._save_settings()

    def _pick_file(self, widget: QtWidgets.QLineEdit):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", str(BASE_DIR))
        if path:
            widget.setText(path)
            self._save_settings()

    def _apply_symbol_args(self, cmd: list[str]):
        symbol_set = self.symbol_set_edit.text().strip()
        if symbol_set:
            cmd.extend(["--symbol-set", symbol_set])

        cleaners = [c.strip() for c in self.text_cleaners_edit.text().split(",") if c.strip()]
        for cleaner in cleaners:
            cmd.extend(["--text-cleaners", cleaner])

        if self.include_style_tokens_check.isChecked():
            cmd.append("--include-style-tokens")

        tags = [t.strip() for t in self.style_tags_edit.text().split(",") if t.strip()]
        if tags:
            cmd.extend(["--style-tags", ",".join(tags)])

        if self.strip_style_check.isChecked():
            cmd.append("--strip-style-from-text")

    def _python_exec(self) -> str:
        if self.use_venv_python_check.isChecked():
            return sys.executable
        return self.python_exec_edit.text().strip() or sys.executable

    def _build_precompile_cmd(self) -> list[str]:
        fastpitch_root = Path(self.fastpitch_folder_edit.text()).expanduser()
        script = fastpitch_root / "prepare_dataset.py"
        cmd = [
            self._python_exec(),
            str(script),
            "-d",
            self.dataset_root_edit.text(),
            "--wav-text-filelists",
            self.train_filelist_edit.text(),
            self.val_filelist_edit.text(),
        ]
        if self.extract_mels_check.isChecked():
            cmd.append("--extract-mels")
        if self.extract_pitch_check.isChecked():
            cmd.append("--extract-pitch")

        cmd.extend(["-b", str(self.prep_batch_spin.value())])
        cmd.extend(["--n-workers", str(self.prep_workers_spin.value())])
        self._apply_symbol_args(cmd)
        return cmd

    def _build_train_cmd(self) -> list[str]:
        fastpitch_root = Path(self.fastpitch_folder_edit.text()).expanduser()
        script = fastpitch_root / "train.py"
        torchrun_exec = self.torchrun_exec_edit.text().strip() or "torchrun"
        cmd = [
            torchrun_exec,
            f"--nproc_per_node={self.nproc_spin.value()}",
            f"--master_port={self.master_port_spin.value()}",
            str(script),
            "--cuda",
            "--epochs",
            str(self.epochs_spin.value()),
            "--epochs-per-checkpoint",
            str(self.epc_spin.value()),
            "-bs",
            str(self.train_batch_spin.value()),
            "-lr",
            str(self.lr_spin.value()),
            "-d",
            self.dataset_root_edit.text(),
            "--num-workers",
            str(self.num_workers_spin.value()),
            "--prefetch-factor",
            str(self.prefetch_spin.value()),
            "--training-files",
            self.train_filelist_edit.text(),
            "--validation-files",
            self.val_filelist_edit.text(),
            "-o",
            self.ckpt_dir_edit.text(),
        ]

        if self.load_mel_check.isChecked():
            cmd.append("--load-mel-from-disk")
        if self.load_pitch_check.isChecked():
            cmd.append("--load-pitch-from-disk")

        self._apply_symbol_args(cmd)
        return cmd

    def _update_command_preview(self, precompile: bool):
        cmd = self._build_precompile_cmd() if precompile else self._build_train_cmd()
        quoted = " ".join(shlex.quote(part) for part in cmd)
        self.command_preview.setPlainText(quoted)

    def _attach_worker(self, cmd: list[str]):
        if self.process_worker and self.process_worker.isRunning():
            self.train_log.appendPlainText("Another process is already running. Stop it first.")
            return False

        self.process_worker = ProcessWorker(cmd)
        self.process_worker.log_line.connect(self.train_log.appendPlainText)
        self.process_worker.finished.connect(self.on_process_finished)
        self.process_worker.start()
        return True

    def start_precompile(self):
        cmd = self._build_precompile_cmd()
        self._update_command_preview(True)
        self._save_settings()
        self.train_log.appendPlainText("Starting precompile…")
        self._attach_worker(cmd)

    def start_training(self):
        cmd = self._build_train_cmd()
        self._update_command_preview(False)
        self._save_settings()
        self.train_log.appendPlainText("Starting training…")
        self._attach_worker(cmd)

    def stop_process(self):
        if self.process_worker and self.process_worker.isRunning():
            self.train_log.appendPlainText("Stopping process…")
            self.process_worker.stop()
        else:
            self.train_log.appendPlainText("No running process to stop.")

    def on_process_finished(self, exit_code: int):
        self.train_log.appendPlainText(f"Process finished with code {exit_code}")

    def open_output_folder(self):
        path = Path(self.ckpt_dir_edit.text()).expanduser()
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _load_settings(self):
        self.fastpitch_edit.setText(self.settings.value("fastpitch", ""))
        self.hifigan_edit.setText(self.settings.value("hifigan", ""))
        self.hifigan_cfg_edit.setText(self.settings.value("hifigan_cfg", ""))

        # FIX: load phrases file (-i)
        self.input_file_edit.setText(self.settings.value("input_file", ""))

        saved_output = self.settings.value("output", "")
        if saved_output:
            self.output_edit.setText(saved_output)

        self.cuda_check.setChecked(self.settings.value("cuda", self.cuda_check.isChecked(), type=bool))
        self.amp_check.setChecked(self.settings.value("amp", self.amp_check.isChecked(), type=bool))
        self.save_mels_check.setChecked(self.settings.value("save_mels", False, type=bool))
        self.pitch_flat_check.setChecked(self.settings.value("pitch_flat", False, type=bool))
        self.pitch_invert_check.setChecked(self.settings.value("pitch_invert", False, type=bool))
        self.pitch_custom_check.setChecked(self.settings.value("pitch_custom", False, type=bool))

        self.pace_spin.setValue(float(self.settings.value("pace", self.pace_spin.value())))
        self.batch_spin.setValue(int(self.settings.value("batch", self.batch_spin.value())))
        self.pitch_amplify_spin.setValue(float(self.settings.value("pitch_amplify", self.pitch_amplify_spin.value())))
        self.pitch_shift_spin.setValue(float(self.settings.value("pitch_shift", self.pitch_shift_spin.value())))

        # Training tab
        self.dataset_root_edit.setText(
            self.settings.value("dataset_root", self.dataset_root_edit.text())
        )
        self.train_filelist_edit.setText(
            self.settings.value("train_filelist", self.train_filelist_edit.text())
        )
        self.val_filelist_edit.setText(
            self.settings.value("val_filelist", self.val_filelist_edit.text())
        )
        self.ckpt_dir_edit.setText(self.settings.value("ckpt_dir", self.ckpt_dir_edit.text()))
        self.fastpitch_folder_edit.setText(
            self.settings.value("fastpitch_folder", self.fastpitch_folder_edit.text())
        )
        self.python_exec_edit.setText(
            self.settings.value("python_exec", self.python_exec_edit.text())
        )
        self.use_venv_python_check.setChecked(
            self.settings.value("use_venv_python", True, type=bool)
        )
        self.torchrun_exec_edit.setText(
            self.settings.value("torchrun_exec", self.torchrun_exec_edit.text())
        )
        self.extract_mels_check.setChecked(
            self.settings.value("extract_mels", self.extract_mels_check.isChecked(), type=bool)
        )
        self.extract_pitch_check.setChecked(
            self.settings.value("extract_pitch", self.extract_pitch_check.isChecked(), type=bool)
        )
        self.prep_batch_spin.setValue(int(self.settings.value("prep_batch", self.prep_batch_spin.value())))
        self.prep_workers_spin.setValue(
            int(self.settings.value("prep_workers", self.prep_workers_spin.value()))
        )
        self.nproc_spin.setValue(int(self.settings.value("nproc", self.nproc_spin.value())))
        self.master_port_spin.setValue(int(self.settings.value("master_port", self.master_port_spin.value())))
        self.epochs_spin.setValue(int(self.settings.value("epochs", self.epochs_spin.value())))
        self.epc_spin.setValue(int(self.settings.value("epc", self.epc_spin.value())))
        self.train_batch_spin.setValue(
            int(self.settings.value("train_batch", self.train_batch_spin.value()))
        )
        self.lr_spin.setValue(float(self.settings.value("lr", self.lr_spin.value())))
        self.num_workers_spin.setValue(
            int(self.settings.value("num_workers", self.num_workers_spin.value()))
        )
        self.prefetch_spin.setValue(int(self.settings.value("prefetch", self.prefetch_spin.value())))
        self.load_mel_check.setChecked(
            self.settings.value("load_mel", self.load_mel_check.isChecked(), type=bool)
        )
        self.load_pitch_check.setChecked(
            self.settings.value("load_pitch", self.load_pitch_check.isChecked(), type=bool)
        )
        self.symbol_set_edit.setText(self.settings.value("symbol_set", self.symbol_set_edit.text()))
        self.text_cleaners_edit.setText(
            self.settings.value("text_cleaners", self.text_cleaners_edit.text())
        )
        self.include_style_tokens_check.setChecked(
            self.settings.value(
                "include_style_tokens",
                self.include_style_tokens_check.isChecked(),
                type=bool,
            )
        )
        self.style_tags_edit.setText(
            self.settings.value("style_tags", self.style_tags_edit.text())
        )
        self.strip_style_check.setChecked(
            self.settings.value("strip_style", self.strip_style_check.isChecked(), type=bool)
        )

    def _save_settings(self):
        self.settings.setValue("fastpitch", self.fastpitch_edit.text())
        self.settings.setValue("hifigan", self.hifigan_edit.text())
        self.settings.setValue("hifigan_cfg", self.hifigan_cfg_edit.text())

        # FIX: save phrases file (-i)
        self.settings.setValue("input_file", self.input_file_edit.text())

        self.settings.setValue("output", self.output_edit.text())
        self.settings.setValue("cuda", self.cuda_check.isChecked())
        self.settings.setValue("amp", self.amp_check.isChecked())
        self.settings.setValue("save_mels", self.save_mels_check.isChecked())
        self.settings.setValue("pitch_flat", self.pitch_flat_check.isChecked())
        self.settings.setValue("pitch_invert", self.pitch_invert_check.isChecked())
        self.settings.setValue("pitch_custom", self.pitch_custom_check.isChecked())
        self.settings.setValue("pace", self.pace_spin.value())
        self.settings.setValue("batch", self.batch_spin.value())
        self.settings.setValue("pitch_amplify", self.pitch_amplify_spin.value())
        self.settings.setValue("pitch_shift", self.pitch_shift_spin.value())
        self.settings.setValue("dataset_root", self.dataset_root_edit.text())
        self.settings.setValue("train_filelist", self.train_filelist_edit.text())
        self.settings.setValue("val_filelist", self.val_filelist_edit.text())
        self.settings.setValue("ckpt_dir", self.ckpt_dir_edit.text())
        self.settings.setValue("fastpitch_folder", self.fastpitch_folder_edit.text())
        self.settings.setValue("python_exec", self.python_exec_edit.text())
        self.settings.setValue("use_venv_python", self.use_venv_python_check.isChecked())
        self.settings.setValue("torchrun_exec", self.torchrun_exec_edit.text())
        self.settings.setValue("extract_mels", self.extract_mels_check.isChecked())
        self.settings.setValue("extract_pitch", self.extract_pitch_check.isChecked())
        self.settings.setValue("prep_batch", self.prep_batch_spin.value())
        self.settings.setValue("prep_workers", self.prep_workers_spin.value())
        self.settings.setValue("nproc", self.nproc_spin.value())
        self.settings.setValue("master_port", self.master_port_spin.value())
        self.settings.setValue("epochs", self.epochs_spin.value())
        self.settings.setValue("epc", self.epc_spin.value())
        self.settings.setValue("train_batch", self.train_batch_spin.value())
        self.settings.setValue("lr", self.lr_spin.value())
        self.settings.setValue("num_workers", self.num_workers_spin.value())
        self.settings.setValue("prefetch", self.prefetch_spin.value())
        self.settings.setValue("load_mel", self.load_mel_check.isChecked())
        self.settings.setValue("load_pitch", self.load_pitch_check.isChecked())
        self.settings.setValue("symbol_set", self.symbol_set_edit.text())
        self.settings.setValue("text_cleaners", self.text_cleaners_edit.text())
        self.settings.setValue("include_style_tokens", self.include_style_tokens_check.isChecked())
        self.settings.setValue("style_tags", self.style_tags_edit.text())
        self.settings.setValue("strip_style", self.strip_style_check.isChecked())

    def closeEvent(self, event):  # noqa: N802,D401
        """Persist settings when the window is closed."""
        self._save_settings()
        super().closeEvent(event)

    def on_finished(self, success: bool, message: str):
        self.log(message)
        if success:
            QtWidgets.QMessageBox.information(self, "Done", message)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", message)


def _run_cli(argv: Sequence[str]) -> bool:
    """Run the pipeline in a headless CLI mode when explicitly requested.

    IMPORTANT: We only enter CLI mode when the user passes --cli.
    This prevents accidental argparse errors when you simply run: `python run.py`
    (which should launch the GUI).

    Returns True if CLI mode handled execution, False otherwise.
    """

    if "--cli" not in argv:
        return False

    parser = argparse.ArgumentParser(
        description="QWPitch + QWGAN CLI (GUI is the default; pass --cli to use this mode)",
        allow_abbrev=False,
    )
    parser.add_argument("--cli", action="store_true", help="Run without the GUI")
    parser.add_argument("-i", "--input", required=True, help="Text file with one line per utterance")
    parser.add_argument("--fastpitch", required=True, help="Path to QWPitch checkpoint")
    parser.add_argument("--hifigan", required=True, help="Path to QWGAN checkpoint")
    parser.add_argument("--hifigan-config", required=True, help="Path to QWGAN config JSON")
    parser.add_argument("-o", "--output", default=str(BASE_DIR / "inference"), help="Output folder root")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--amp", action="store_true", help="Use AMP for QWPitch")
    parser.add_argument("--pace", type=float, default=1.0, help="Speaking pace multiplier")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for QWPitch")
    parser.add_argument("--save-mels", action="store_true", help="Save intermediate mel .npy files")
    parser.add_argument("--pitch-flatten", action="store_true", help="Flatten pitch contours")
    parser.add_argument("--pitch-invert", action="store_true", help="Invert pitch contours")
    parser.add_argument("--pitch-amplify", type=float, default=1.0, help="Amplify pitch variation")
    parser.add_argument("--pitch-shift", type=float, default=0.0, help="Shift pitch in Hz")
    parser.add_argument(
        "--pitch-custom",
        action="store_true",
        help="Enable custom transform from third_party/fastpitch/pitch_transform.py",
    )

    args = parser.parse_args(argv[1:])

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input text file not found: {input_path}")

    lines = [ln.strip() for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Input text file is empty after stripping blank lines")

    cmudict.initialize(str(BASE_DIR / "third_party/fastpitch/cmudict/cmudict-0.7b"), keep_ambiguous=True)

    print(f"Tokenizer: {summarize_tokenizer()}", file=sys.stderr)

    pitch = PitchSettings(
        flatten=args.pitch_flatten,
        invert=args.pitch_invert,
        amplify=args.pitch_amplify,
        shift_hz=args.pitch_shift,
        custom=args.pitch_custom,
    )

    wavs = run_pipeline(
        text_lines=lines,
        fastpitch_ckpt=Path(args.fastpitch),
        hifigan_ckpt=Path(args.hifigan),
        hifigan_cfg=Path(args.hifigan_config),
        output_root=Path(args.output),
        use_cuda=args.cuda,
        use_amp=args.amp,
        pace=args.pace,
        pitch=pitch,
        save_mels=args.save_mels,
        batch_size=args.batch_size,
        progress=lambda msg: print(msg, file=sys.stderr),
    )

    for path in wavs:
        print(path)

    return True


def main():
    if _run_cli(sys.argv):
        return

    # Ensure Qt doesn't try to interpret CLI-only arguments
    qt_argv = [sys.argv[0]]

    app = QtWidgets.QApplication(qt_argv)
    window = MainWindow()
    window.resize(800, 650)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
