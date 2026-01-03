"""Minimal FastPitch + HiFi-GAN GUI launcher.

This GUI keeps only the FastPitch (text -> mel) and HiFi-GAN (mel -> wav)
parts of the stack. It shells out to the existing CLI utilities so the core
training/inference code stays unchanged.
"""

import datetime
import subprocess
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

BASE_DIR = Path(__file__).parent.resolve()


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
    ):
        super().__init__()
        self.text = text
        self.fastpitch_ckpt = Path(fastpitch_ckpt).resolve()
        self.hifigan_ckpt = Path(hifigan_ckpt).resolve()
        self.hifigan_config = Path(hifigan_config).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.use_cuda = use_cuda

    def run(self):
        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            inference_root = BASE_DIR / "inference"
            phrases_dir = inference_root / "phrases"
            mel_dir = inference_root / "mels" / stamp
            wav_dir = self.output_dir

            phrases_dir.mkdir(parents=True, exist_ok=True)
            mel_dir.mkdir(parents=True, exist_ok=True)
            wav_dir.mkdir(parents=True, exist_ok=True)

            text_path = phrases_dir / f"gui_{stamp}.txt"
            text_path.write_text(self.text.strip() + "\n", encoding="utf-8")
            self.progress.emit(f"Wrote phrase file: {text_path.relative_to(BASE_DIR)}")

            fastpitch_cmd = [
                sys.executable,
                str(BASE_DIR / "third_party/fastpitch/inference.py"),
                "-i",
                str(text_path),
                "-o",
                str(mel_dir),
                "--save-mels",
                "--fastpitch",
                str(self.fastpitch_ckpt),
                "--batch-size",
                "1",
            ]
            if self.use_cuda:
                fastpitch_cmd.append("--cuda")

            self.progress.emit("Running FastPitch inference…")
            subprocess.run(fastpitch_cmd, check=True, cwd=BASE_DIR)

            mel_files = sorted(mel_dir.glob("*.npy"))
            if not mel_files:
                raise RuntimeError("FastPitch did not produce any mel files.")
            mel_path = mel_files[0]
            self.progress.emit(f"Generated mel: {mel_path.relative_to(BASE_DIR)}")

            wav_path = wav_dir / f"{mel_path.stem}.wav"
            hifigan_cmd = [
                sys.executable,
                str(BASE_DIR / "tools/mel_to_wav_hifigan.py"),
                "--mel",
                str(mel_path),
                "--hifigan",
                str(self.hifigan_ckpt),
                "--config",
                str(self.hifigan_config),
                "--out",
                str(wav_path),
            ]
            if self.use_cuda:
                hifigan_cmd.append("--cuda")

            self.progress.emit("Running HiFi-GAN vocoder…")
            subprocess.run(hifigan_cmd, check=True, cwd=BASE_DIR)

            self.finished.emit(True, f"Saved {wav_path.relative_to(BASE_DIR)}")
        except subprocess.CalledProcessError as exc:
            self.finished.emit(False, f"Command failed: {' '.join(exc.cmd)}")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FastPitch + HiFi-GAN GUI")
        self.worker = None

        self.fastpitch_edit = QtWidgets.QLineEdit()
        self.hifigan_edit = QtWidgets.QLineEdit()
        self.hifigan_cfg_edit = QtWidgets.QLineEdit()
        self.output_edit = QtWidgets.QLineEdit(str((BASE_DIR / "inference" / "wavs").resolve()))
        self.text_edit = QtWidgets.QPlainTextEdit(
            "FastPitch and HiFi-GAN now drive this repository."
        )
        self.cuda_check = QtWidgets.QCheckBox("Use CUDA if available")
        self.cuda_check.setChecked(True)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)

        synth_btn = QtWidgets.QPushButton("Synthesize")
        synth_btn.clicked.connect(self.start_synthesis)

        layout = QtWidgets.QFormLayout()
        for label, widget in (
            self._row("FastPitch checkpoint", self.fastpitch_edit, self.browse_model),
            self._row("HiFi-GAN checkpoint", self.hifigan_edit, self.browse_hifigan),
            self._row("HiFi-GAN config", self.hifigan_cfg_edit, self.browse_hifigan_cfg),
            self._row("Output directory", self.output_edit, self.browse_output),
        ):
            layout.addRow(label, widget)

        layout.addRow("Text", self.text_edit)
        layout.addRow(self.cuda_check)
        layout.addRow(synth_btn)
        layout.addRow("Log", self.log_box)
        self.setLayout(layout)

    def _row(self, label, widget, handler):
        h = QtWidgets.QHBoxLayout()
        h.addWidget(widget)
        btn = QtWidgets.QPushButton("Browse")
        btn.clicked.connect(handler)
        h.addWidget(btn)
        row = QtWidgets.QWidget()
        row.setLayout(h)
        return QtWidgets.QLabel(label), row

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
        output_dir = (
            Path(self.output_edit.text()).resolve()
            if self.output_edit.text()
            else (BASE_DIR / "inference" / "wavs").resolve()
        )

        missing = [p for p in (fastpitch, hifigan, hifigan_cfg) if not p.is_file()]
        if missing:
            self.log(f"Missing paths: {', '.join(str(m) for m in missing)}")
            return

        text = self.text_edit.toPlainText().strip()
        if not text:
            self.log("Please enter text to synthesize.")
            return

        self.worker = SynthWorker(
            text=text,
            fastpitch_ckpt=fastpitch,
            hifigan_ckpt=hifigan,
            hifigan_config=hifigan_cfg,
            output_dir=output_dir,
            use_cuda=self.cuda_check.isChecked(),
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
    window.resize(700, 500)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
