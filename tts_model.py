# tts_model.py

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
if not hasattr(np, 'complex'): np.complex = complex  # numpy<->librosa compat
if not hasattr(np, 'float'): np.float = float        # numpy<->librosa compat
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.utils as nn_utils
from datetime import datetime

# ---------------------------------------------------------------------
# Text and Audio Preprocessing Functions
# ---------------------------------------------------------------------

def text_to_sequence(text, char_to_idx):
    """Convert text to a sequence of integer indices using char_to_idx."""
    text = text.lower()
    seq = [char_to_idx.get(ch, char_to_idx.get(" ")) for ch in text]
    return torch.LongTensor(seq)

def load_wav(wav_path, sample_rate=22050):
    y, sr = librosa.load(wav_path, sr=sample_rate)
    return y

def compute_mel_spectrogram(y, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
    """
    Compute a mel spectrogram from a waveform and normalize it.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels)
    ref_value = np.max(mel)
    mel_db = librosa.power_to_db(mel, ref=ref_value)
    mel_db = np.clip(mel_db, a_min=-100, a_max=0)
    normalized_mel = (mel_db + 100) / 100.0
    return torch.FloatTensor(normalized_mel)

# ---------------------------------------------------------------------
# Dataset and Collate Function
# ---------------------------------------------------------------------
class TTSDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Pad text sequences
    seqs = [item[0] for item in batch]
    max_seq_len = max([s.size(0) for s in seqs])
    padded_seqs = [torch.cat([s, torch.zeros(max_seq_len - s.size(0), dtype=s.dtype)])
                   if s.size(0) < max_seq_len else s for s in seqs]
    batch_seqs = torch.stack(padded_seqs)

    # Pad mel spectrograms (assumed shape: [n_mels, time])
    mels = [item[1] for item in batch]
    max_mel_len = max([mel.size(1) for mel in mels])
    padded_mels = [torch.cat([mel, torch.zeros(mel.size(0), max_mel_len - mel.size(1))], dim=1)
                   if mel.size(1) < max_mel_len else mel for mel in mels]
    batch_mels = torch.stack(padded_mels)
    return batch_seqs, batch_mels

# ---------------------------------------------------------------------
# Model Modules
# ---------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, conv_channels, kernel_size, num_conv_layers, lstm_hidden):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim if i == 0 else conv_channels, conv_channels,
                          kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Dropout(0.5)
            ) for i in range(num_conv_layers)
        ])
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, bidirectional=True, batch_first=True)
    def forward(self, x):
        # x: [B, seq_length]
        x = self.embedding(x).transpose(1, 2)  # [B, embedding_dim, seq_length]
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)  # [B, seq_length, conv_channels]
        outputs, _ = self.lstm(x)  # [B, seq_length, 2*lstm_hidden]
        return outputs

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [B, decoder_dim]
        # encoder_outputs: [B, seq_length, encoder_dim]
        dec = self.W_decoder(decoder_hidden).unsqueeze(1)  # [B, 1, attention_dim]
        enc = self.W_encoder(encoder_outputs)  # [B, seq_length, attention_dim]
        scores = self.V(torch.tanh(enc + dec))  # [B, seq_length, 1]
        scores = scores.squeeze(-1)  # [B, seq_length]
        attn_weights = F.softmax(scores, dim=1)  # [B, seq_length]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, encoder_dim]
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, mel_channels, decoder_dim, encoder_dim, attention_dim):
        super(Decoder, self).__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.lstm = nn.LSTMCell(mel_channels + encoder_dim, decoder_dim)
        self.linear = nn.Linear(decoder_dim + encoder_dim, mel_channels)
        self.gate = nn.Linear(decoder_dim + encoder_dim, 1)
    def forward(self, encoder_outputs, target_mel=None, teacher_forcing_ratio=0.5, max_len=200, gate_threshold=0.5):
        batch_size = encoder_outputs.size(0)
        decoder_dim = self.lstm.hidden_size
        device = encoder_outputs.device
        mel_input = torch.zeros(batch_size, target_mel.size(1) if target_mel is not None else 80, device=device)
        outputs = []
        hidden = torch.zeros(batch_size, decoder_dim, device=device)
        cell = torch.zeros(batch_size, decoder_dim, device=device)
        
        for t in range(max_len):
            context, _ = self.attention(hidden, encoder_outputs)
            lstm_input = torch.cat([mel_input, context], dim=1)
            hidden, cell = self.lstm(lstm_input, (hidden, cell))
            concat = torch.cat([hidden, context], dim=1)
            output = self.linear(concat)
            outputs.append(output.unsqueeze(1))
            
            # Compute gate output from the same concatenated vector.
            gate_output = self.gate(concat)  # shape: [B, 1]
            
            # If we're in inference mode (no teacher forcing), check if the average gate exceeds threshold.
            if target_mel is None and teacher_forcing_ratio == 0:
                if torch.mean(gate_output) > gate_threshold:
                    break
            
            if target_mel is not None and t < target_mel.size(2) and torch.rand(1).item() < teacher_forcing_ratio:
                mel_input = target_mel[:, :, t]
            else:
                mel_input = output
        outputs = torch.cat(outputs, dim=1)  # shape: [B, T, mel_channels]
        return outputs.transpose(1, 2)  # shape: [B, mel_channels, T]

class PostNet(nn.Module):
    def __init__(self, mel_channels, postnet_channels, kernel_size, num_convs):
        super(PostNet, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = mel_channels if i == 0 else postnet_channels
            out_channels = mel_channels if i == num_convs - 1 else postnet_channels
            activation = nn.Tanh() if i < num_convs - 1 else nn.Identity()
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(out_channels),
                    activation,
                    nn.Dropout(0.5)
                )
            )
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class Tacotron2(nn.Module):
    def __init__(self, vocab_size, mel_channels=80, embedding_dim=256,
                 encoder_conv_channels=512, encoder_conv_layers=3, encoder_lstm_hidden=256,
                 decoder_dim=512, attention_dim=128, postnet_channels=512,
                 postnet_kernel_size=5, postnet_conv_layers=5, max_len=200):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, encoder_conv_channels,
                               kernel_size=5, num_conv_layers=encoder_conv_layers,
                               lstm_hidden=encoder_lstm_hidden)
        self.decoder = Decoder(mel_channels, decoder_dim, encoder_lstm_hidden*2, attention_dim)
        self.postnet = PostNet(mel_channels, postnet_channels, postnet_kernel_size, postnet_conv_layers)
        self.max_len = max_len
    def forward(self, text_seq, target_mel=None, teacher_forcing_ratio=0):
        encoder_outputs = self.encoder(text_seq)
        decoder_outputs = self.decoder(encoder_outputs, target_mel, teacher_forcing_ratio, max_len=self.max_len)
        mel_outputs = decoder_outputs
        mel_outputs_post = mel_outputs + self.postnet(mel_outputs)
        return mel_outputs, mel_outputs_post

def update_metadata_file(audio_filename, transcript, metadata_path=os.path.join("datasets", "my_dataset", "metadata.csv")):
    """
    Updates the metadata CSV file in LJ Speech format.
    Writes a single line for the given audio file and transcript.
    
    Returns the LJ code generated.
    """
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Read existing lines to determine the next LJ code
    existing_lines = []
    if os.path.isfile(metadata_path):
        with open(metadata_path, mode="r", encoding="utf-8") as f:
            existing_lines = [line.strip() for line in f if line.strip()]
    
    max_index = 0
    for line in existing_lines:
        try:
            code = line.split("|")[0]  # e.g. "LJ001-0001"
            parts = code[2:].split("-")
            if len(parts) == 2:
                num = int(parts[0]) * 10000 + int(parts[1])
                if num > max_index:
                    max_index = num
        except Exception:
            pass

    next_num = max_index + 1
    new_code = "LJ" + f"{next_num // 10000:03d}" + "-" + f"{next_num % 10000:04d}"
    new_line = f"{new_code}|{transcript}|{transcript}"
    with open(metadata_path, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_file.write(new_line + "\n")
    print(f"Updated metadata file: {metadata_path} with {new_line}")
    return new_code

# ---------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------
def train_model(config_path, output_path, progress_callback=None):
    # Create logs folder if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    sample_rate = config["audio"].get("sample_rate", 22050)
    n_mels = config["audio"].get("num_mels", 80)
    n_fft = config["audio"].get("fft_size", 1024)
    hop_length = config["audio"].get("hop_length", 256)
    
    dataset = config["datasets"][0]
    # For LJ Speech, assume metadata is in datasets/my_dataset/metadata.csv
    metadata_path = os.path.join(dataset["path"], "metadata.csv")
    wav_dir = os.path.join(dataset["path"], "wavs")
    
    vocab_set = set()
    data = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            file_id, transcription = parts[0], parts[1]
            vocab_set.update(list(transcription.lower()))
            wav_file = os.path.join(wav_dir, file_id + ".wav")
            if os.path.exists(wav_file):
                y = load_wav(wav_file, sample_rate)
                mel = compute_mel_spectrogram(y, sample_rate, n_fft, hop_length, n_mels)
                data.append((transcription, mel))
    if len(data) == 0:
        print("No data found. Check your metadata and WAV files.")
        return
    
    vocab = sorted(list(vocab_set))
    if " " not in vocab:
        vocab.append(" ")
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    vocab_size = len(vocab)
    
    processed_data = []
    for text, mel in data:
        seq = text_to_sequence(text, char_to_idx)
        processed_data.append((seq, mel))
    
    dataset_obj = TTSDataset(processed_data)
    data_loader = DataLoader(dataset_obj, batch_size=config.get("batch_size", 4),
                             shuffle=True, num_workers=config.get("num_workers", 4), collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tacotron2(vocab_size, mel_channels=n_mels, max_len=400)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    num_epochs = config.get("epochs", 50)
    
    total_iterations = num_epochs * len(data_loader)
    current_iter = 0
    print(f"Starting training on {len(processed_data)} samples for {num_epochs} epochs with batch size {config.get('batch_size',4)}.")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        count = 0
        for batch_seqs, batch_mels in data_loader:
            batch_seqs = batch_seqs.to(device)
            batch_mels = batch_mels.to(device)
            optimizer.zero_grad()
            mel_pred, mel_post = model(batch_seqs, target_mel=batch_mels, teacher_forcing_ratio=config.get("teacher_forcing_ratio", 1.0))
            T_pred = mel_pred.size(2)
            loss = criterion(mel_pred, batch_mels[:, :, :T_pred])
            loss.backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]))
            print(f"Before Clipping: Gradient Norm = {total_norm:.4f}")
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_norm_after = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]))
            print(f"After Clipping: Gradient Norm = {total_norm_after:.4f}")
            optimizer.step()
            total_loss += loss.item()
            count += 1
            current_iter += 1
            progress_percent = int(current_iter / total_iterations * 100)
            if progress_callback:
                progress_callback(progress_percent, f"Epoch {epoch+1}/{num_epochs} - Batch {count}/{len(data_loader)} - Loss: {loss.item():.4f}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {count}/{len(data_loader)}, Loss: {loss.item():.6f}, Avg Loss: {total_loss/count:.6f}, LR: {current_lr:.8f}")
            with open(log_filename, "a") as log_file:
                log_file.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"Epoch {epoch+1}/{num_epochs}, Batch {count}/{len(data_loader)}, "
                    f"Loss: {loss.item():.6f}, Avg Loss: {total_loss/count:.6f}, LR: {current_lr:.8f}\n"
                )
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{num_epochs} Completed. Average Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)
    
    os.makedirs(output_path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "char_to_idx": char_to_idx,
        "vocab": vocab,
        "n_mels": n_mels
    }
    model_file = os.path.join(output_path, "tacotron2_model.pth")
    torch.save(checkpoint, model_file)
    print("Training complete. Model saved to:", model_file)

# ---------------------------------------------------------------------
# Synthesis Function
# ---------------------------------------------------------------------
def synthesize(text, output_file, model_path, config_path=None, use_cuda=True):
    import warnings
    warnings.filterwarnings("ignore", message="pytorch_quantization module not found")
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        checkpoint["model_state_dict"] = new_state_dict
    char_to_idx = checkpoint.get("char_to_idx", {})
    vocab = checkpoint.get("vocab", [])
    n_mels = checkpoint.get("n_mels", 80)
    vocab_size = len(vocab) if vocab else 50
    
    # Instantiate the single-speaker model
    model = Tacotron2(vocab_size, mel_channels=n_mels, max_len=300).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    seq = text_to_sequence(text, char_to_idx).unsqueeze(0).to(device)
    
    # Load WaveGlow from TorchHub for vocoding
    with torch.no_grad():
        mel_pred, mel_post = model(seq, teacher_forcing_ratio=0.0)

    audio = None
    try:
        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub',
                                  'nvidia_waveglow', model_math='fp32',
                                  verbose=True, map_location=device).to(device)
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow.eval()
        with torch.no_grad():
            audio = waveglow.infer(mel_post).cpu().numpy()[0]
    except Exception as e:
        # Fallback vocoder
        import librosa
        import numpy as np
        from librosa.feature.inverse import mel_to_stft
        from librosa import griffinlim
        mel = mel_post.squeeze(0).cpu().numpy()  # [n_mels, T]
        S = mel_to_stft(mel, sr=22050, n_fft=1024, hop_length=256, power=1.0)
        audio = griffinlim(S, n_iter=60, hop_length=256, win_length=1024)

    sf.write(output_file, audio, 22050)
    return output_file

# ---------------------------------------------------------------------
# Adaptation Function (Fine Tuning)
# --------------------------------------------------------------------- 
def adapt_speaker(model, adaptation_data, char_to_idx, num_epochs=5, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for transcript, mel in adaptation_data:
            text_seq = text_to_sequence(transcript, char_to_idx).unsqueeze(0).to(device)
            mel = mel.unsqueeze(0).to(device)
            optimizer.zero_grad()
            mel_pred, _ = model(text_seq, target_mel=mel, teacher_forcing_ratio=1.0)
            T_pred = mel_pred.size(2)
            loss = criterion(mel_pred, mel[:, :, :T_pred])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Adaptation Epoch {epoch+1}/{num_epochs}: Loss = {total_loss:.6f}")
    model.eval()
    return model

