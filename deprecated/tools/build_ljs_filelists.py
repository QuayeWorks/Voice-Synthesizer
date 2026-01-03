import os
import sys
import argparse
import random

# --- Add project root to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from third_party.tacotron2.text import text_to_sequence


def build_filelists(ljs_root, out_dir,
                    train_name="ljs_train_clean.txt",
                    val_name="ljs_val_clean.txt",
                    val_fraction=0.01,
                    cleaners=("english_cleaners",)):
    metadata_path = os.path.join(ljs_root, "metadata.csv")
    wav_dir = os.path.join(ljs_root, "wavs")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
    if not os.path.exists(wav_dir):
        raise FileNotFoundError(f"wavs dir not found at {wav_dir}")

    os.makedirs(out_dir, exist_ok=True)

    entries = []
    skipped_missing_wav = 0
    skipped_empty_seq = 0

    print(f"[build] reading {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # LJS format: ID|raw_text|normalized_text
            parts = line.split("|")
            if len(parts) < 3:
                continue
            utt_id = parts[0]
            raw_text = parts[1]
            norm_text = parts[2]

            wav_path = os.path.join(wav_dir, utt_id + ".wav")
            if not os.path.exists(wav_path):
                skipped_missing_wav += 1
                continue

            # Use the normalized text (matches NVIDIA)
            text = norm_text.strip()
            if not text:
                continue

            # Run through text_to_sequence with english_cleaners
            seq = text_to_sequence(text, list(cleaners))
            if len(seq) == 0:
                skipped_empty_seq += 1
                continue

            # Use absolute paths (you've been using those already)
            wav_abs = os.path.abspath(wav_path)
            entries.append(f"{wav_abs}|{text}")

    print(f"[build] total usable entries: {len(entries)}")
    print(f"[build] skipped missing wavs: {skipped_missing_wav}")
    print(f"[build] skipped empty sequences after cleaners: {skipped_empty_seq}")

    # Shuffle and split into train/val
    random.shuffle(entries)
    n_total = len(entries)
    n_val = max(1, int(val_fraction * n_total))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    train_path = os.path.join(out_dir, train_name)
    val_path = os.path.join(out_dir, val_name)

    with open(train_path, "w", encoding="utf-8") as f:
        for e in train_entries:
            f.write(e + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for e in val_entries:
            f.write(e + "\n")

    print(f"[build] wrote {len(train_entries)} lines to {train_path}")
    print(f"[build] wrote {len(val_entries)} lines to {val_path}")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ljs-root",
        type=str,
        required=True,
        help="Path to LJSpeech-1.1 directory (containing metadata.csv and wavs/)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="filelists",
        help="Directory to write train/val filelists into.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.01,
        help="Fraction of data to use for validation (default 0.01 = 1%%).",
    )
    args = parser.parse_args()

    build_filelists(
        ljs_root=args.ljs_root,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
