"""PyQt5 GUI for FastPitch (text->mel) + HiFi-GAN (mel->wav).

This GUI keeps the modern pipeline only (no Tacotron2/WaveGlow).
- FastPitch produces mels directly from text.
- HiFi-GAN vocodes those mels into WAV files.

Features
--------
- Model selection for FastPitch checkpoint, HiFi-GAN generator + config.
- CUDA + AMP toggles (AMP only affects FastPitch when CUDA is on).
- Pace + pitch transform controls for FastPitch.
- Multi-line text input (one line = one output wav) with batch processing.
- Optional mel saving alongside WAVs.
- Output directory picker with /mels + /wavs subfolders.
- Progress log panel with per-step updates.
- Global naming to avoid overwritten outputs.
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from contextlib import nullcontext
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch
from scipy.io.wavfile import write

# Prefer the offscreen Qt plugin by default so the app can run in environments
# without the xcb dependencies. Users can override by setting
# QT_QPA_PLATFORM explicitly (e.g., to "xcb" when a display is available).
if not os.getenv("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt5 import QtCore, QtWidgets

# FastPitch + HiFi-GAN imports
from third_party.fastpitch import inference as fp_infer
from third_party.fastpitch.common.text import cmudict
from third_party.hifigan.env import AttrDict
from third_party.hifigan.models import Generator

BASE_DIR = Path(__file__).parent.resolve()


@dataclass
class PitchSettings:
    flatten: bool = False
    invert: bool = False
    amplify: float = 1.0
    shift_hz: float = 0.0
    custom: bool = False


@dataclass
class FastPitchOutput:
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
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser = fp_infer.parse_args(parser)
    args, unk_args = parser.parse_known_args([])
    args.amp = use_amp
    model = fp_infer.load_and_setup_model(
        "FastPitch",
        parser,
        str(ckpt),
        use_amp,
        device,
        unk_args=unk_args,
        forward_is_infer=True,
        ema=args.ema,
        jitable=args.torchscript,
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
) -> List[FastPitchOutput]:
    """Run FastPitch on provided text lines and return mel outputs."""

    if len(texts) != len(stems):
        raise ValueError("texts and stems must have the same length")

    if progress:
        progress("Loading FastPitch…")

    if any(len(t.strip()) == 0 for t in texts):
        raise ValueError("Blank lines are not allowed in the input text")

    amp_enabled = use_amp and device.type == "cuda"
    model = _prepare_fastpitch_model(checkpoint, device, amp_enabled)

    if progress:
        progress("Preparing input batches…")

    fields = {"text": list(texts), "output": list(stems)}
    batches = fp_infer.prepare_input_sequence(
        fields,
        device,
        symbol_set="english_basic",
        text_cleaners=["english_cleaners_v2"],
        batch_size=batch_size,
        dataset=None,
        load_mels=False,
        p_arpabet=0.0,
    )

    pitch_transform = _build_pitch_transform(pitch)

    gen_kw = {
        "pace": pace,
        "speaker": 0,
        "pitch_tgt": None,
        "pitch_transform": pitch_transform,
    }

    outputs: List[FastPitchOutput] = []
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
            outputs.append(FastPitchOutput(stem=stem, mel=mel_trimmed, mel_path=mel_path))

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
    mels: Iterable[FastPitchOutput],
    generator_path: Path,
    config_path: Path,
    out_dir: Path,
    device: torch.device,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Path]:
    """Convert mels to wavs using HiFi-GAN."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Loading HiFi-GAN…")

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
        progress(f"FastPitch finished. Vocoding {len(fp_outputs)} mel(s)…")

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
        self.setWindowTitle("FastPitch + HiFi-GAN GUI")
        self.worker: Optional[SynthWorker] = None

        # Inputs
        self.fastpitch_edit = QtWidgets.QLineEdit()
        self.hifigan_edit = QtWidgets.QLineEdit()
        self.hifigan_cfg_edit = QtWidgets.QLineEdit()
        default_out = (BASE_DIR / "inference").resolve()
        self.output_edit = QtWidgets.QLineEdit(str(default_out))

        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setPlaceholderText("Enter one phrase per line")

        self.cuda_check = QtWidgets.QCheckBox("Use CUDA if available")
        self.cuda_check.setChecked(torch.cuda.is_available())
        self.amp_check = QtWidgets.QCheckBox("Use AMP (FastPitch)")
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
            ("FastPitch checkpoint", self.fastpitch_edit, self.browse_model),
            ("HiFi-GAN checkpoint", self.hifigan_edit, self.browse_hifigan),
            ("HiFi-GAN config", self.hifigan_cfg_edit, self.browse_hifigan_cfg),
            ("Output directory", self.output_edit, self.browse_output),
        ):
            layout.addRow(label, self._row(widget, handler))

        layout.addRow("Text", self.text_edit)
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
        self.setLayout(layout)

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

    def browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select FastPitch checkpoint", str(BASE_DIR))
        if path:
            self.fastpitch_edit.setText(path)

    def browse_hifigan(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select HiFi-GAN checkpoint", str(BASE_DIR))
        if path:
            self.hifigan_edit.setText(path)

    def browse_hifigan_cfg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select HiFi-GAN config", str(BASE_DIR))
        if path:
            self.hifigan_cfg_edit.setText(path)

    def browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", str(BASE_DIR))
        if path:
            self.output_edit.setText(path)

    def log(self, message: str):
        self.log_box.appendPlainText(message)

    def start_synthesis(self):
        if self.worker is not None and self.worker.isRunning():
            self.log("Synthesis already running…")
            return

        fastpitch = Path(self.fastpitch_edit.text()).resolve()
        hifigan = Path(self.hifigan_edit.text()).resolve()
        hifigan_cfg = Path(self.hifigan_cfg_edit.text()).resolve()
        output_dir = Path(self.output_edit.text()).resolve() if self.output_edit.text() else (BASE_DIR / "inference").resolve()

        missing = [p for p in (fastpitch, hifigan, hifigan_cfg) if not p.is_file()]
        if missing:
            self.log(f"Missing paths: {', '.join(str(m) for m in missing)}")
            return

        text = self.text_edit.toPlainText()
        if not text.strip():
            self.log("Please enter text to synthesize.")
            return

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

    def on_finished(self, success: bool, message: str):
        self.log(message)
        if success:
            QtWidgets.QMessageBox.information(self, "Done", message)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", message)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 650)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
