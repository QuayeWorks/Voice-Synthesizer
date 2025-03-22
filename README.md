**QuayeWorks Voice Synthesizer – Neural Text-to-Speech System with Whisper Integration
**
**Transform Text, Voice, and Waveforms into Intelligent Speech**

The QuayeWorks Voice Synthesizer is a robust, GUI-based Python application designed to train, fine-tune, and deploy custom neural text-to-speech (TTS) models. Built on Tacotron2 architecture and powered by NVIDIA WaveGlow for neural vocoding, this synthesizer enables users to convert text into lifelike speech, record and transcribe new voice data using OpenAI's Whisper, and visualize spectrograms during model debugging.

Whether you're a researcher building a custom dataset, an AI developer training a personal voice model, or an enthusiast exploring synthetic speech, this tool provides a streamlined end-to-end workflow—from data recording and transcription to real-time synthesis and fine-tuning.

**Key Features & Benefits**
End-to-End Neural Speech Synthesis

- Based on a custom implementation of Tacotron2 with trainable encoder-decoder-attention-postnet pipeline.

- Real-time synthesis using NVIDIA’s WaveGlow vocoder.

**Whisper-Integrated Voice Recording**

- Records microphone input and transcribes it into text using OpenAI's Whisper ASR model.

- Automatically segments audio and updates metadata files in LJ Speech format.

**Dataset Management**

Automatically builds training datasets from recorded or uploaded audio files.

- Seamless management of transcripts, metadata, and audio segments.

**Training & Fine-Tuning Capabilities**

- Trains custom voice models from scratch using mel spectrogram loss.

- Supports single-speaker adaptation for fine-tuning on new voice data.

- Includes adjustable hyperparameters (batch size, learning rate, data workers, cuDNN toggles).

**Interactive TTS GUI (PyQt5)
**
- Input text and instantly generate, play, and save synthesized audio.

- Record and transcribe audio directly in the app.

- Load, visualize, and inspect model architecture and spectrogram outputs.

- Real-time logs for training and adaptation tasks.

**Advanced Debugging & Visualization**

- Standalone mel spectrogram visualizer for Tacotron2 output.

- Spectrogram overlays using librosa.display and matplotlib.

**Technical Stack & Dependencies**
- Framework: Python 3.x with PyTorch

- Model Architecture: Tacotron2 + WaveGlow

- Audio: librosa, soundfile, PyAudio

- ASR: OpenAI Whisper (GPU-accelerated)

- UI: PyQt5

- Visualization: matplotlib

- Training Logs: Auto-generated per session

- File Management: Supports WAV/MP3, segmenting, renaming, and metadata tracking

**Ideal Use Cases**
- Custom AI Voice Generation

- TTS Research and Prototyping

- Audio Dataset Creation and Expansion

- AI Education and Demonstrations

- Voice Cloning and Single-Speaker Adaptation

**Recomendations**
- Hardware: atleast TWO NVIDIA CUDA Supported GPUS. i.e 2x1080TI+ capable GPU's
- Software: Install all libraries, and use \VENV\  when running.
- Training: Atleast 24Hr's of transcribed audio data with > 50 Epochs.

**Disclaimer**
This synthesizer is intended for educational, research, and lawful development purposes only. The user assumes full responsibility for generated content and any downstream applications. The developers disclaim liability for misuse, unethical use, or any breach of legal or ethical guidelines involving this software.

Let Me know of any bugs.
