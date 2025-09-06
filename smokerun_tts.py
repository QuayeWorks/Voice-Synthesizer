#!/usr/bin/env python3
"""
smokerun_tts.py — Minimal end-to-end smoke test for your TTS project.

What it does (in order):
  1) Loads your config (default: temp_config.json if present, else config.json)
  2) Creates a tiny "smoke_subset" dataset (2 items) from your main dataset
  3) Writes a temporary config pointing to the subset with epochs=1, batch_size=2
  4) Trains once (calls tts_model.train_model) and saves a checkpoint
  5) Synthesizes a short sentence using the new checkpoint and writes smoke_sample.wav

This does NOT judge audio quality — it only verifies the pipeline runs end-to-end.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

# Helpful for newest numpy + librosa combos (no-ops if attrs already exist)
import numpy as np
if not hasattr(np, "complex"): np.complex = complex
if not hasattr(np, "float"): np.float = float

import tts_model

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def build_smoke_subset(src_dataset: Path, dst_dataset: Path, max_items: int = 2) -> int:
    """Create a tiny dataset copy with up to max_items entries that have matching wavs."""
    meta_src = src_dataset / "metadata.csv"
    wavs_src = src_dataset / "wavs"
    assert meta_src.exists(), f"metadata.csv not found at {meta_src}"
    assert wavs_src.exists(), f"wavs dir not found at {wavs_src}"
    dst_dataset.mkdir(parents=True, exist_ok=True)
    (dst_dataset / "wavs").mkdir(parents=True, exist_ok=True)

    kept = 0
    lines_out = []
    with open(meta_src, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            parts = s.split("|")
            file_id = parts[0]
            wav_src = wavs_src / f"{file_id}.wav"
            if wav_src.exists():
                # copy wav
                wav_dst = (dst_dataset / "wavs" / f"{file_id}.wav")
                shutil.copy2(wav_src, wav_dst)
                lines_out.append(s)
                kept += 1
                if kept >= max_items:
                    break
    assert kept > 0, "No valid (wav + metadata) pairs found to build a subset."
    with open(dst_dataset / "metadata.csv", "w", encoding="utf-8") as f:
        for line in lines_out:
            f.write(line + "\n")
    return kept

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to base config (default: temp_config.json or config.json)")
    parser.add_argument("--subset_items", type=int, default=2, help="How many items to include in the smoke subset")
    parser.add_argument("--output_dir", type=str, default="datasets/output_smoke", help="Where to write smoke outputs")
    parser.add_argument("--use_cuda", action="store_true", help="Enable CUDA if available")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.", help="Text to synthesize")
    args = parser.parse_args()

    root = Path(".").resolve()

    # Choose base config
    if args.config:
        base_cfg_path = Path(args.config)
    else:
        # Prefer temp_config.json if present, else config.json
        base_cfg_path = Path("temp_config.json") if Path("temp_config.json").exists() else Path("config.json")
    assert base_cfg_path.exists(), f"Config file not found: {base_cfg_path}"

    base_cfg = load_config(base_cfg_path)
    # Get dataset path from config
    datasets = base_cfg.get("datasets") or [{"path": "datasets/my_dataset"}]
    src_dataset = Path(datasets[0]["path"])

    # Build subset
    subset_path = Path("datasets/smoke_subset")
    subset_count = build_smoke_subset(src_dataset, subset_path, max_items=args.subset_items)
    print(f"[smoke] Built subset at {subset_path} with {subset_count} item(s).")

    # Create a temp config for smoke run
    smoke_cfg = dict(base_cfg)  # shallow copy
    smoke_cfg.setdefault("teacher_forcing_ratio", 1.0)
    smoke_cfg["epochs"] = 1
    smoke_cfg["batch_size"] = max(2, min(8, subset_count))  # keep tiny
    smoke_cfg["lr"] = base_cfg.get("lr", 0.0015)  # leave as-is if present
    smoke_cfg["datasets"] = [{"path": str(subset_path)}]

    smoke_cfg_path = Path("datasets/smoke_subset_config.json")
    save_json(smoke_cfg, smoke_cfg_path)
    print(f"[smoke] Wrote temp config: {smoke_cfg_path}")

    # Train once
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[smoke] Starting training for 1 epoch...")
    tts_model.train_model(str(smoke_cfg_path), str(out_dir))
    model_path = out_dir / "tacotron2_model.pth"
    assert model_path.exists(), f"Model checkpoint not found at {model_path}"
    print(f"[smoke] Training complete: {model_path}")

    # Synthesize a short line
    wav_out = out_dir / "smoke_sample.wav"
    print("[smoke] Synthesizing a short test sentence...")
    tts_model.synthesize(args.text, str(wav_out), str(model_path), config_path=str(smoke_cfg_path), use_cuda=args.use_cuda)
    assert wav_out.exists(), f"Synthesis failed: {wav_out} not created"
    print(f"[smoke] Synthesis OK: {wav_out}")

    print("\n✅ Smoke run completed successfully.")
    print(f"   • Dataset subset:   {subset_path} ({subset_count} items)")
    print(f"   • Config used:      {smoke_cfg_path}")
    print(f"   • Model checkpoint: {model_path}")
    print(f"   • Output WAV:       {wav_out}")

if __name__ == "__main__":
    main()
