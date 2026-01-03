
import os, argparse, csv
from pathlib import Path
import numpy as np

try:
    import torch, torchaudio
    HAVE_TORCHAUDIO = True
except Exception:
    HAVE_TORCHAUDIO = False

try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

try:
    import soundfile as sf
except Exception:
    sf = None


def load_audio_mono_22050(path):
    target_sr = 22050
    if HAVE_TORCHAUDIO:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.squeeze(0).cpu().numpy(), target_sr
    elif HAVE_LIBROSA:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y, sr
    else:
        if sf is None: raise RuntimeError("Need audio backend")
        y, sr = sf.read(path, always_2d=False)
        if y.ndim == 2: y = y.mean(axis=1)
        if sr != target_sr:
            xp = np.linspace(0, 1, len(y), endpoint=False)
            xq = np.linspace(0, 1, int(len(y)*target_sr/sr), endpoint=False)
            y = np.interp(xq, xp, y).astype(np.float32)
            sr = target_sr
        return y, sr


def save_wav(path, y, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if HAVE_TORCHAUDIO:
        T = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        torchaudio.save(path, T, sr)
    elif sf is not None:
        sf.write(path, y.astype(np.float32), sr)
    else:
        raise RuntimeError("Need torchaudio or soundfile to write audio.")


def peak_normalize_dbfs(y, dbfs=-3.0):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        gain = (10.0 ** (dbfs/20.0)) / peak
        y = y * gain
    return y


def truncate_silence(y, sr, threshold_db=-20.0, min_silence_s=0.5, truncate_to_s=0.5):
    frame = int(0.02*sr)
    hop = int(0.01*sr)
    if len(y) < frame: return y.copy()
    rms = np.sqrt(np.convolve(y*y, np.ones(frame), 'valid')/frame + 1e-12)
    idxs = np.arange(0, len(rms)-1, hop, dtype=int)
    rms = rms[idxs]
    thr = 10.0**(threshold_db/20.0)
    below = rms < thr
    min_frames = int(min_silence_s / (hop/sr))
    keep_frames = int(truncate_to_s / (hop/sr))

    segments = []
    run = 0
    for i, b in enumerate(below):
        if b: run += 1
        else:
            if run >= min_frames:
                end = i; start = i-run
                segments.append((start,end))
            run = 0
    if run >= min_frames:
        end = len(below); start = end-run
        segments.append((start,end))

    if not segments:
        return y.copy()

    y_out = []
    cursor = 0
    for (s, e) in segments:
        s_samp = s*hop
        e_samp = min(len(y), e*hop + frame)
        y_out.append(y[cursor:s_samp])
        keep_len = int(keep_frames*hop)
        y_out.append(y[s_samp:min(e_samp, s_samp+keep_len)])
        cursor = e_samp
    y_out.append(y[cursor:])
    return np.concatenate(y_out) if len(y_out)>1 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    wavs = root / "wavs"
    out_wavs = root / "wavs_trimmed"
    out_meta = root / "metadata_trimmed.csv"

    os.makedirs(out_wavs, exist_ok=True)

    transcripts = {}
    meta = root / "metadata.csv"
    if meta.exists():
        with open(meta, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="|")
            for row in r:
                if not row: continue
                transcripts[Path(row[0]).stem] = row[1] if len(row)>1 else ""

    rows = []
    files = sorted([p for p in wavs.glob("*.wav")])
    for p in files:
        try:
            y, sr = load_audio_mono_22050(str(p))
            y = peak_normalize_dbfs(y, -3.0)
            y = truncate_silence(y, sr, threshold_db=-20.0, min_silence_s=0.5, truncate_to_s=0.5)
            pad = int(0.06*sr); y = np.pad(y, (pad,pad))
            save_wav(str(out_wavs/p.name), y, sr)
            txt = transcripts.get(p.stem, "")
            rows.append([p.stem, txt, txt])
        except Exception as e:
            print("Failed:", p, e)

    with open(out_meta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        for row in rows: w.writerow(row)

    print(f"Done. Wrote {len(rows)} items to {out_wavs} and {out_meta}")


if __name__ == "__main__":
    main()
