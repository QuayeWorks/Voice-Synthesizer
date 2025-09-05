import sys
import os
import wave
import json
import csv
import time
import pyaudio
import soundfile as sf
import numpy as np
if not hasattr(np, 'complex'): np.complex = complex  # numpy<->librosa compat
if not hasattr(np, 'float'): np.float = float        # numpy<->librosa compat
import speech_recognition as sr
import warnings
import torch
import librosa
import shutil
import whisper

from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QTextEdit, QListWidget,
    QProgressBar, QMessageBox, QFileDialog, QComboBox, QSpinBox, QCheckBox, QDoubleSpinBox, QLineEdit, QGridLayout
)

warnings.filterwarnings("ignore", message="pytorch_quantization module not found")

import tts_model  # Your custom TTS model module (single-speaker Tacotron2 + update_metadata_file)

# -----------------------------
# Options Tab
# -----------------------------
class OptionsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        # Batch Size Option
        batch_layout = QHBoxLayout()
        batch_label = QLabel("Batch Size:")
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setMaximum(64)
        self.batch_spin.setValue(60)
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_spin)
        layout.addLayout(batch_layout)
        
        # DataLoader Workers Option
        worker_layout = QHBoxLayout()
        worker_label = QLabel("DataLoader Workers:")
        self.worker_spin = QSpinBox()
        self.worker_spin.setMinimum(0)
        self.worker_spin.setMaximum(16)
        self.worker_spin.setValue(8)
        worker_layout.addWidget(worker_label)
        worker_layout.addWidget(self.worker_spin)
        layout.addLayout(worker_layout)
        
        # cuDNN Benchmarking Option
        self.cudnn_checkbox = QCheckBox("Enable cuDNN Benchmarking")
        self.cudnn_checkbox.setChecked(False)
        layout.addWidget(self.cudnn_checkbox)
        
        # Learning Rate Option
        lr_layout = QHBoxLayout()
        lr_label = QLabel("Learning Rate:")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setMinimum(0.000001)
        self.lr_spin.setMaximum(1.0)
        self.lr_spin.setValue(0.0015)
        lr_layout.addWidget(lr_label)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        # Apply Options Button
        self.apply_button = QPushButton("Apply Options")
        layout.addWidget(self.apply_button)
        self.setLayout(layout)

# -----------------------------
# Recorder Thread (using Whisper)
# -----------------------------
class RecorderThread(QThread):
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)  # <-- Add this line
    
    def __init__(self, transcript_text, audio_folder, audio_count):
        super().__init__()
        self.transcript_text = transcript_text  # fallback if ASR fails
        self.audio_folder = audio_folder
        self.audio_count = audio_count
        self.is_recording = True
        
    def run(self):
        try:
            # Initialize the Whisper model
            device_whisper = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model = whisper.load_model("medium", device=device_whisper)
            
            
            chunk = 1024
            sample_format = pyaudio.paInt16
            channels = 1
            target_sr = 22050
            input_sr = 44100  # safer for most devices
            p = pyaudio.PyAudio()
            stream = p.open(format=sample_format, channels=channels, rate=input_sr, input=True, frames_per_buffer=chunk)

            frames = []
            self.recording_started.emit()
            while self.is_recording:
                data = stream.read(chunk)
                frames.append(data)
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save the full recording with a temporary base name
            temp_base = f"recording_{self.audio_count+1}"
            temp_audio_file = f"{temp_base}.wav"
            full_audio_path = os.path.join(self.audio_folder, temp_audio_file)
            wf = wave.open(full_audio_path, "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(input_sr)
            wf.writeframes(b"".join(frames))
            wf.close()
            
            # Split the full recording into 4-second segments
            y, sr = librosa.load(full_audio_path, sr=input_sr)
            y = librosa.resample(y, orig_sr=input_sr, target_sr=target_sr)
            sr = target_sr
            segment_length = 4 * sr  # ~4 seconds in samples
            num_segments = int(np.ceil(len(y) / segment_length))
            print(f"Splitting recording into {num_segments} segment(s).")
            
            for i in range(num_segments):
                start_sample = i * segment_length
                end_sample = min((i+1) * segment_length, len(y))
                segment_audio = y[start_sample:end_sample]
                
                # Save segment
                temp_segment_file = f"{temp_base}_seg_{i+1}.wav"
                segment_path = os.path.join(self.audio_folder, temp_segment_file)
                sf.write(segment_path, segment_audio, sr)
                self.progress_updated.emit(int(((i + 0.25) / num_segments) * 100), f"Segment {i+1}: file saved.")
                #print(f"[DEBUG] Segment {i+1} written to {segment_path}, exists? {os.path.exists(segment_path)}")
                
                # Short delay to ensure OS flushes the file
                #import time
                #time.sleep(0.2)
                
                # Double-check existence
                print(f"[DEBUG] Checking again, segment {i+1} path: {segment_path}, exists? {os.path.exists(segment_path)}")
                
                # Whisper transcription
                print(f"[DEBUG] Starting Whisper on segment {i+1}")
                #time.sleep(0.2)
                if not os.path.exists(segment_path):
                    print(f"[DEBUG] File {segment_path} is missing before transcription!")
                    continue  # skip or handle error

                # Transcribe
                #print(f"[DEBUG] Calling whisper_model.transcribe for segment {i+1}")
                try:
                    result = whisper_model.transcribe(segment_path)
                    print(f"[DEBUG] Transcription result for segment {i+1}: {result['text']}")
                except Exception as transcribe_error:
                    print(f"[DEBUG] Error during transcription for segment {i+1}: {transcribe_error}")
                    result = {"text": ""}
                self.progress_updated.emit(int(((i + 0.5) / num_segments) * 100), f"Segment {i+1}: Transcribed.")
                # Another short delay after reading
                #time.sleep(0.2)
                # Check if file still exists
                #print(f"[DEBUG] After Whisper, segment {i+1}, path: {segment_path}, exists? {os.path.exists(segment_path)}")
                
                # Fallback logic for transcript
                transcript_to_use = result["text"] if result["text"].strip() != "" else self.transcript_text
                
                # Update metadata
                new_lj_code = tts_model.update_metadata_file(temp_segment_file, transcript_to_use)
                new_audio_file = f"{new_lj_code}.wav"
                new_audio_path = os.path.join(self.audio_folder, new_audio_file)
                self.progress_updated.emit(int(((i + 0.75) / num_segments) * 100), f"Segment {i+1}: Metadata Updated.")
                # Final rename
                #print(f"[DEBUG] Attempting rename: {segment_path} -> {new_audio_path}")
                if os.path.exists(segment_path):
                    #time.sleep(0.2)
                    try:
                        os.rename(segment_path, new_audio_path)
                        #print(f"[DEBUG] Segment {i+1} renamed to {new_audio_file}")
                    except Exception as rename_err:
                        print(f"[DEBUG] Rename error for segment {i+1}: {rename_err}")
                else:
                    print(f"[DEBUG] Segment file not found at rename step: {segment_path}")
                self.progress_updated.emit(int(((i + 1) / num_segments) * 100), f"Segment {i+1} Complete! ")
            
            # Optionally remove the full temporary recording
            if os.path.exists(full_audio_path):
                os.remove(full_audio_path)
            
            self.recording_stopped.emit("Recording processed into segments.")
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        self.is_recording = False

# -----------------------------
# Training Thread
# -----------------------------
class TrainingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    training_completed = pyqtSignal(bool)
    
    def __init__(self, config_path, output_path):
        super().__init__()
        self.config_path = config_path
        self.output_path = output_path
        
    def run(self):
        try:
            self.progress_updated.emit(0, "Starting training...")
            tts_model.train_model(self.config_path, self.output_path, progress_callback=self.progress_updated.emit)
            self.progress_updated.emit(100, "Training complete!")
            self.training_completed.emit(True)
        except Exception as e:
            self.progress_updated.emit(0, f"Error: {e}")
            self.training_completed.emit(False)

# -----------------------------
# Main App Window (Single-Speaker)
# -----------------------------
class VoiceSynthesizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Synthesizer App - Single Speaker w/ Whisper")
        self.setGeometry(100, 100, 900, 650)
        
        self.audio_folder = os.path.join("datasets", "my_dataset", "wavs")
        print("Audio folder:", self.audio_folder)
        if not os.path.exists(self.audio_folder):
            os.makedirs(self.audio_folder)
        self.metadata_file = os.path.join("datasets", "my_dataset", "metadata.csv")
        
        self.recording = False
        self.media_player = QMediaPlayer()
        self.recorder_thread = None
        self.training_thread = None
        self.training_completed = False
        
        self.tts_model = None
        self.char_to_idx = None
        self.n_mels = None
        self.model_file = os.path.join("datasets", "output", "tacotron2_model.pth")
        
        # Default Options
        self.config_batch_size = 60
        self.config_num_workers = 8
        self.config_cudnn_benchmark = False
        self.config_learning_rate = 0.0015
        
        self.record_icon = QIcon(os.path.join("icons", "record.png"))
        self.pause_icon = QIcon(os.path.join("icons", "pause.png"))
        
        self.init_ui()
        self.load_existing_model()
    
    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.training_tab = QWidget()
        self.tts_tab = QWidget()
        self.adaptation_tab = QWidget()  # For fine-tuning if desired
        self.transcript_tab = QWidget()
        self.options_tab = OptionsTab()
        
        self.options_tab.apply_button.clicked.connect(self.apply_options)
        
        self.tabs.addTab(self.training_tab, "Training & Transcription")
        self.tabs.addTab(self.tts_tab, "Text-to-Speech")
        self.tabs.addTab(self.adaptation_tab, "Adaptation")
        self.tabs.addTab(self.transcript_tab, "Transcripts")
        self.tabs.addTab(self.options_tab, "Options")
        
        self.init_training_tab()
        self.init_tts_tab()
        self.init_adaptation_tab()
        self.init_transcript_tab()
    
    def apply_options(self):
        self.config_batch_size = self.options_tab.batch_spin.value()
        self.config_num_workers = self.options_tab.worker_spin.value()
        self.config_cudnn_benchmark = self.options_tab.cudnn_checkbox.isChecked()
        self.config_learning_rate = self.options_tab.lr_spin.value()
        torch.backends.cudnn.benchmark = self.config_cudnn_benchmark
        print(f"Options Applied: Batch Size = {self.config_batch_size}, "
              f"Workers = {self.config_num_workers}, cuDNN = {self.config_cudnn_benchmark}, "
              f"LR = {self.config_learning_rate}")
        QMessageBox.information(self, "Options",
            f"Options Applied:\nBatch Size: {self.config_batch_size}\n"
            f"Workers: {self.config_num_workers}\n"
            f"cuDNN: {self.config_cudnn_benchmark}\n"
            f"LR: {self.config_learning_rate}")
    
    def init_training_tab(self):
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        
        self.upload_audio_button = QPushButton("Upload Audio Files")
        self.upload_audio_button.clicked.connect(self.upload_audio_files)
        
        self.record_button = QPushButton()
        self.record_button.setIcon(self.record_icon)
        self.record_button.setIconSize(QSize(32,32))
        self.record_button.clicked.connect(self.record_audio)
        
        self.remove_button = QPushButton("Remove Selected Audio")
        self.remove_button.clicked.connect(self.remove_selected_audio)
        
        self.upload_transcript_button = QPushButton("Upload Transcript")
        self.upload_transcript_button.clicked.connect(self.upload_transcript)
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.start_training)
        
        self.transcribe_button = QPushButton("Transcribe Audio Files")
        self.transcribe_button.clicked.connect(self.transcribe_audio_files)
        
        button_layout.addWidget(self.upload_audio_button)
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.upload_transcript_button)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.transcribe_button)
        
        self.audio_list = QListWidget()
        self.audio_list.itemDoubleClicked.connect(self.play_selected_audio)
        
        layout.addLayout(button_layout)
        layout.addWidget(QLabel("Audio Files:"))
        layout.addWidget(self.audio_list)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.training_status = QLabel("Training not started.")
        
        layout.addWidget(QLabel("Training Progress:"))
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.training_status)
        
        self.training_tab.setLayout(layout)
        self.load_audio_files()
    
    def init_tts_tab(self):
        layout = QVBoxLayout()
        input_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to synthesize...")
        
        self.generate_button = QPushButton("Generate Speech")
        self.generate_button.setEnabled(False)  # Enabled after model is loaded
        self.generate_button.clicked.connect(self.generate_speech)
        
        self.play_button = QPushButton("Play Audio")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_synthesized_audio)
        
        self.save_button = QPushButton("Save Audio")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_synthesized_audio)
        
        input_layout.addWidget(QLabel("Text Input:"))
        input_layout.addWidget(self.text_input)
        
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(input_layout)
        layout.addLayout(button_layout)
        self.tts_tab.setLayout(layout)
    
    def init_adaptation_tab(self):
        # If you want to allow fine-tuning on single speaker data
        layout = QGridLayout()
        folder_label = QLabel("Adaptation Folder:")
        self.folder_lineedit = QLineEdit()
        self.folder_lineedit.setReadOnly(True)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_adaptation_folder)
        
        epochs_label = QLabel("Adaptation Epochs:")
        self.adapt_epochs_spin = QSpinBox()
        self.adapt_epochs_spin.setMinimum(1)
        self.adapt_epochs_spin.setMaximum(100)
        self.adapt_epochs_spin.setValue(5)
        
        lr_label = QLabel("Adaptation LR:")
        self.adapt_lr_spin = QDoubleSpinBox()
        self.adapt_lr_spin.setDecimals(6)
        self.adapt_lr_spin.setSingleStep(0.0001)
        self.adapt_lr_spin.setMinimum(0.000001)
        self.adapt_lr_spin.setMaximum(1.0)
        self.adapt_lr_spin.setValue(0.0001)
        
        fine_tune_button = QPushButton("Fine-Tune Model")
        fine_tune_button.clicked.connect(self.fine_tune_model)
        
        self.adapt_status = QLabel("Adaptation not started.")
        
        layout.addWidget(folder_label, 0, 0)
        layout.addWidget(self.folder_lineedit, 0, 1)
        layout.addWidget(browse_button, 0, 2)
        
        layout.addWidget(epochs_label, 1, 0)
        layout.addWidget(self.adapt_epochs_spin, 1, 1)
        
        layout.addWidget(lr_label, 2, 0)
        layout.addWidget(self.adapt_lr_spin, 2, 1)
        
        layout.addWidget(fine_tune_button, 3, 0, 1, 3)
        layout.addWidget(self.adapt_status, 4, 0, 1, 3)
        
        self.adaptation_tab.setLayout(layout)
    
    def init_transcript_tab(self):
        layout = QVBoxLayout()
        load_button = QPushButton("Load Transcripts")
        load_button.clicked.connect(self.load_transcripts)
        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        
        layout.addWidget(load_button)
        layout.addWidget(QLabel("Transcript Content:"))
        layout.addWidget(self.transcript_text)
        self.transcript_tab.setLayout(layout)
    
    def browse_adaptation_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Adaptation Folder", os.getcwd())
        if folder:
            self.folder_lineedit.setText(folder)
    
    def load_transcripts(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                self.transcript_text.setPlainText("".join(lines))
            else:
                self.transcript_text.setPlainText("No transcripts available.")
        else:
            self.transcript_text.setPlainText("Metadata file not found.")
    
    def load_audio_files(self):
        self.audio_list.clear()
        if os.path.exists(self.audio_folder):
            for filename in os.listdir(self.audio_folder):
                if filename.endswith(".wav"):
                    self.audio_list.addItem(filename)
    
    def upload_audio_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Upload Audio Files", "", "Audio Files (*.wav *.mp3);;All Files (*)", options=options)
        if files:
            for file in files:
                base_name = os.path.basename(file)
                dest_path = os.path.join(self.audio_folder, base_name)
                if not os.path.exists(dest_path):
                    try:
                        with open(file, "rb") as f_src, open(dest_path, "wb") as f_dest:
                            f_dest.write(f_src.read())
                        self.audio_list.addItem(base_name)
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Could not copy file {file}: {e}")
                else:
                    QMessageBox.information(self, "Info", f"File {base_name} already exists.")
    
    def remove_selected_audio(self):
        selected_items = self.audio_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Select an audio file to remove.")
            return
        for item in selected_items:
            filename = item.text()
            path = os.path.join(self.audio_folder, filename)
            if os.path.exists(path):
                os.remove(path)
                self.audio_list.takeItem(self.audio_list.row(item))
            else:
                QMessageBox.warning(self, "Warning", f"File not found: {path}")
    
    def upload_transcript(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Upload Transcript", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file:
            base_name = os.path.basename(file)
            dest_path = self.metadata_file
            try:
                with open(file, "r", encoding="utf-8") as f_src:
                    transcript = f_src.read().strip()
                tts_model.update_metadata_file(base_name, transcript, metadata_path=dest_path)
                QMessageBox.information(self, "Info", f"Transcript {base_name} appended to metadata.")
                self.load_transcripts()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not upload transcript: {e}")
   
    def update_recording_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.training_status.setText(message)

    def record_audio(self):
        if not self.recording:
            self.recording = True
            audio_count = len([f for f in os.listdir(self.audio_folder) if f.endswith(".wav")])
            self.recorder_thread = RecorderThread("", self.audio_folder, audio_count)
            self.recorder_thread.recording_started.connect(self.on_recording_started)
            self.recorder_thread.recording_stopped.connect(self.on_recording_stopped)
            self.recorder_thread.error_occurred.connect(self.on_recording_error)
            self.recorder_thread.progress_updated.connect(self.update_recording_progress)  # <-- Add this line
            self.recorder_thread.start()
        else:
            self.record_button.setIcon(self.record_icon)
            self.record_button.setIconSize(QSize(32,32))
            self.recording = False
            if self.recorder_thread is not None:
                self.recorder_thread.stop()
                self.recorder_thread.wait()
                self.recorder_thread = None
    
    def on_recording_started(self):
        print("Recording started.")
        self.record_button.setIcon(self.pause_icon)
        self.record_button.setIconSize(QSize(32,32))
    
    def on_recording_stopped(self, message):
        print("Recording stopped and processed:", message)
        self.record_button.setIcon(self.record_icon)
        self.recording = False
        self.recorder_thread = None
        self.load_audio_files()
        self.load_transcripts()
    
    def on_recording_error(self, error_message):
        QMessageBox.critical(self, "Recording Error", f"Error during recording:\n{error_message}")
        self.record_button.setIcon(self.record_icon)
        self.recording = False
        self.recorder_thread = None
    
    def play_selected_audio(self, item):
        file_path = os.path.join(self.audio_folder, item.text())
        url = QUrl.fromLocalFile(file_path)
        content = QMediaContent(url)
        self.media_player.setMedia(content)
        self.media_player.play()
    
    def start_training(self):
        config_path = "config.json"
        output_path = os.path.join("datasets", "output")
        # Load config, update with options
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["batch_size"] = self.config_batch_size
        config["num_workers"] = self.config_num_workers
        config["lr"] = self.config_learning_rate
        temp_config_path = "temp_config.json"
        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        self.training_thread = TrainingThread(temp_config_path, output_path)
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.start()
        self.train_button.setEnabled(False)
        self.training_status.setText("Training in progress...")
        QMessageBox.information(self, "Info", "Model training started. This may take a while.")
    
    def update_training_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.training_status.setText(message)
    
    def on_training_completed(self, success):
        if success:
            QMessageBox.information(self, "Info", "Model training completed successfully.")
            self.generate_button.setEnabled(True)
            self.load_existing_model()
        else:
            QMessageBox.critical(self, "Error", "Model training failed.")
        self.train_button.setEnabled(True)
        self.training_status.setText("Training completed.")
    
    def transcribe_audio_files(self):
        # If you want to automatically transcribe existing .wav files, you could implement
        # a separate logic or rely on the new RecorderThread logic. 
        # For now, just show a message.
        QMessageBox.information(self, "Transcription", "This feature not implemented in Whisper version.")
    
    def generate_speech(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Enter text to synthesize.")
            return
        output_file = "synthesized_audio.wav"
        try:
            tts_model.synthesize(text, output_file, model_path=self.model_file, use_cuda=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Synthesis failed: {e}")
            return
        self.play_button.setEnabled(True)
        self.save_button.setEnabled(True)
        QMessageBox.information(self, "Info", "Speech synthesis completed.")
    
    def play_synthesized_audio(self):
        file_path = "synthesized_audio.wav"
        if os.path.exists(file_path):
            url = QUrl.fromLocalFile(file_path)
            content = QMediaContent(url)
            self.media_player.setMedia(content)
            self.media_player.play()
        else:
            QMessageBox.critical(self, "Error", "Synthesized audio file not found.")
    
    def save_synthesized_audio(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getSaveFileName(self, "Save Audio File", "", "Audio Files (*.wav);;All Files (*)", options=options)
        if file:
            try:
                os.rename("synthesized_audio.wav", file)
                QMessageBox.information(self, "Info", f"Audio saved to {file}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save audio: {e}")
    
    def load_existing_model(self):
        if os.path.exists(self.model_file):
            print("Loading pre-trained model from", self.model_file)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.model_file, map_location=device)
            state_dict = checkpoint["model_state_dict"]
            # If multiple GPUs were used, handle "module." prefix
            if torch.cuda.device_count() > 1:
                if not list(state_dict.keys())[0].startswith("module."):
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict["module." + k] = v
                    checkpoint["model_state_dict"] = new_state_dict
            else:
                if list(state_dict.keys())[0].startswith("module."):
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict[k[7:]] = v
                    checkpoint["model_state_dict"] = new_state_dict
            
            self.char_to_idx = checkpoint.get("char_to_idx", {})
            self.n_mels = checkpoint.get("n_mels", 80)
            vocab = checkpoint.get("vocab", [])
            vocab_size = len(vocab) if vocab else 50
            
            # Instantiate single-speaker Tacotron2
            self.tts_model = tts_model.Tacotron2(vocab_size, mel_channels=self.n_mels, max_len=400).to(device)
            self.tts_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.tts_model.eval()
            
            self.generate_button.setEnabled(True)
            print("Model loaded successfully.")
        else:
            print("No pre-trained model found. Please train a model first.")
            self.tts_model = None
            # generate_button stays disabled
    
    def fine_tune_model(self):
        # For single-speaker adaptation
        folder = self.folder_lineedit.text().strip()
        if not folder:
            QMessageBox.warning(self, "Warning", "Please select an adaptation folder.")
            return
        if self.tts_model is None:
            QMessageBox.warning(self, "Warning", "No trained model loaded.")
            return
        self.adapt_status.setText("Loading adaptation data...")
        adaptation_data = []
        for filename in os.listdir(folder):
            if filename.endswith(".wav"):
                wav_path = os.path.join(folder, filename)
                txt_path = os.path.splitext(wav_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                    try:
                        y = tts_model.load_wav(wav_path, sample_rate=22050)
                        mel = tts_model.compute_mel_spectrogram(y, sample_rate=22050)
                        adaptation_data.append((transcript, mel))
                    except Exception as e:
                        print(f"Error processing {wav_path}: {e}")
        if not adaptation_data:
            QMessageBox.warning(self, "Warning", "No adaptation data found in the folder.")
            self.adapt_status.setText("No adaptation data found.")
            return
        
        epochs = self.adapt_epochs_spin.value()
        lr = self.adapt_lr_spin.value()
        self.adapt_status.setText("Fine-tuning in progress...")
        QApplication.processEvents()
        
        try:
            adapted_model = tts_model.adapt_speaker(self.tts_model, adaptation_data, char_to_idx=self.char_to_idx, num_epochs=epochs, lr=lr)
            self.tts_model = adapted_model
            self.adapt_status.setText("Adaptation complete.")
            QMessageBox.information(self, "Adaptation", "Model fine-tuning complete.")
        except Exception as e:
            self.adapt_status.setText("Adaptation failed.")
            QMessageBox.critical(self, "Error", f"Adaptation failed: {e}")

def main():
    app = QApplication(sys.argv)
    window = VoiceSynthesizerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()