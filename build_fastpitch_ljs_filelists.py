import os
import random

# Project root = folder containing datasets/, third_party/, etc.
ROOT = os.path.dirname(os.path.abspath(__file__))

# LJSpeech-1.1 location
LJS_ROOT = os.path.join(ROOT, "datasets", "LJSpeech-1.1")
METADATA_PATH = os.path.join(LJS_ROOT, "metadata.csv")
WAV_DIR = os.path.join(LJS_ROOT, "wavs")

# Where FastPitch expects its filelists
FP_FILELIST_DIR = os.path.join(ROOT, "third_party", "fastpitch", "filelists")
os.makedirs(FP_FILELIST_DIR, exist_ok=True)

TRAIN_NAME = "ljs_audio_text_train_filelist.txt"
VAL_NAME   = "ljs_audio_text_val_filelist.txt"
VAL_FRACTION = 0.01  # 1% val


def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"metadata.csv not found at {METADATA_PATH}")
    if not os.path.exists(WAV_DIR):
        raise FileNotFoundError(f"wavs dir not found at {WAV_DIR}")

    entries = []
    skipped_missing_wav = 0
    skipped_malformed = 0
    skipped_empty_text = 0

    print(f"[build] reading {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            # LJS formats:
            #   [id, raw, norm]   (standard)
            #   [id, norm]        (some variants)
            if len(parts) < 2:
                skipped_malformed += 1
                continue

            utt_id = parts[0].strip()
            if len(parts) >= 3:
                text = parts[2].strip()   # normalized text
            else:
                text = parts[1].strip()

            if not utt_id or not text:
                skipped_empty_text += 1
                continue

            wav_path = os.path.join(WAV_DIR, utt_id + ".wav")
            if not os.path.exists(wav_path):
                skipped_missing_wav += 1
                continue

            # FastPitch wants path *relative to dataset root*
            # (we'll tell it dataset-path=LJS_ROOT)
            wav_rel = f"wavs/{utt_id}.wav"
            entries.append(f"{wav_rel}|{text}")

    print(f"[build] total usable entries: {len(entries)}")
    print(f"[build] skipped missing wavs: {skipped_missing_wav}")
    print(f"[build] skipped malformed lines: {skipped_malformed}")
    print(f"[build] skipped empty text: {skipped_empty_text}")

    # Shuffle and split train/val (1% val)
    random.shuffle(entries)
    n_total = len(entries)
    n_val = max(1, int(VAL_FRACTION * n_total))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    train_path = os.path.join(FP_FILELIST_DIR, TRAIN_NAME)
    val_path   = os.path.join(FP_FILELIST_DIR, VAL_NAME)

    # Write with header line: audio|text
    with open(train_path, "w", encoding="utf-8") as ft:
        ft.write("audio|text\n")
        for e in train_entries:
            ft.write(e + "\n")

    with open(val_path, "w", encoding="utf-8") as fv:
        fv.write("audio|text\n")
        for e in val_entries:
            fv.write(e + "\n")

    print(f"[build] wrote {len(train_entries)} lines to {train_path}")
    print(f"[build] wrote {len(val_entries)} lines to {val_path}")


if __name__ == "__main__":
    main()
