
import os, math, argparse, json, csv, re, time
from pathlib import Path
import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

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
    import whisper
    HAVE_WHISPER = True
except Exception:
    HAVE_WHISPER = False


def load_audio_mono(path, target_sr=22050):
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
        if sf is None: raise RuntimeError("Need torchaudio or librosa or soundfile")
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


def group_indices(n, min_size=8, max_size=12):
    if n <= max_size:
        yield (0, n); return
    i = 0
    while i < n:
        remain = n - i
        size = min(max_size, max(min_size, int(round(remain / max(1, math.ceil(remain/10))))))
        size = min(max_size, max(min_size, size))
        yield (i, min(n, i+size))
        i += size


def whisper_segments(path, model_name="small"):
    if not HAVE_WHISPER:
        return None
    m = whisper.load_model(model_name)
    res = m.transcribe(path, language="en")
    segs = res.get("segments", [])
    return [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]} for s in segs]


def next_lj_name_from_folder(wavs_dir):
    used = set([p.name for p in Path(wavs_dir).glob("LJ???-???.wav")])
    a,b = 0,1
    while True:
        nm = f"LJ{a:03d}-{b:03d}.wav"
        if nm not in used:
            used.add(nm); yield nm
        b += 1
        if b>999: b=1; a+=1
        if a>999: raise RuntimeError("LJ label space exhausted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_wav", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--use_whisper", action="store_true")
    ap.add_argument("--threshold_db", type=float, default=-20.0)
    ap.add_argument("--min_silence", type=float, default=0.5)
    ap.add_argument("--truncate_to", type=float, default=0.5)
    args = ap.parse_args()

    ds = Path(args.dataset_root)
    wavs_dir = ds/"wavs"
    meta_path = ds/"metadata.csv"
    os.makedirs(wavs_dir, exist_ok=True)

    y, sr = load_audio_mono(args.input_wav, 22050)
    y = peak_normalize_dbfs(y, -3.0)
    y = truncate_silence(y, sr, args.threshold_db, args.min_silence, args.truncate_to)

    rows = []
    name_iter = next_lj_name_from_folder(wavs_dir)

    if args.use_whisper and HAVE_WHISPER:
        segs = whisper_segments(args.input_wav) or []
        if segs:
            for (a,b) in group_indices(len(segs)):
                st = int(segs[a]["start"]*sr); en = int(segs[b-1]["end"]*sr)
                clip = y[st:en]
                pad = int(0.06*sr); clip = np.pad(clip, (pad,pad))
                clip = peak_normalize_dbfs(clip, -3.0)
                name = next(name_iter)
                save_wav(str(wavs_dir/name), clip, sr)
                text = " ".join([segs[i]["text"].strip() for i in range(a,b)]).strip()
                rows.append([Path(name).stem, text, text])
        else:
            name = next(name_iter); save_wav(str(wavs_dir/name), y, sr)
            rows.append([Path(name).stem, "", ""])
    else:
        name = next(name_iter); save_wav(str(wavs_dir/name), y, sr)
        rows.append([Path(name).stem, "", ""])

    with open(meta_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        for r in rows: w.writerow(r)

    print(f"Wrote {len(rows)} clip(s) to {wavs_dir} and appended to {meta_path}")


if __name__ == "__main__":
    main()
