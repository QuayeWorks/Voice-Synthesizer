from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from fastpitch_config import DEFAULT_CONFIG_PATH, FastPitchProjectConfig, load_fastpitch_config
from third_party.fastpitch import train as fastpitch_train


def _split_metadata(
    metadata: Path, output_dir: Path, val_ratio: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    if not metadata.is_file():
        raise FileNotFoundError(metadata)

    lines = [ln.strip() for ln in metadata.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Metadata file {metadata} is empty.")

    rng = random.Random(seed)
    rng.shuffle(lines)

    split_idx = max(1, int(len(lines) * (1 - val_ratio)))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:] or lines[-1:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_filelist.txt"
    val_path = output_dir / "val_filelist.txt"
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    return [train_path], [val_path]


def _resolve_filelists(files: Sequence[str]) -> List[Path]:
    return [Path(f).expanduser().resolve() for f in files]


def _build_training_cli(
    config: FastPitchProjectConfig,
    output_dir: Path,
    train_files: List[Path],
    val_files: List[Path],
    use_cuda: bool,
    override_epochs: int | None,
    override_batch: int | None,
) -> List[str]:
    tokenizer = config.tokenizer
    training = config.training
    epochs = override_epochs or training.epochs
    batch_size = override_batch or training.batch_size

    args: List[str] = [
        "--output",
        str(output_dir),
        "--dataset-path",
        str(training.dataset_path),
        "--epochs",
        str(epochs),
        "--epochs-per-checkpoint",
        str(training.epochs_per_checkpoint),
        "--learning-rate",
        str(training.learning_rate),
        "--batch-size",
        str(batch_size),
        "--grad-accumulation",
        str(training.grad_accumulation),
        "--warmup-steps",
        str(training.warmup_steps),
        "--dur-predictor-loss-scale",
        str(training.dur_loss_scale),
        "--pitch-predictor-loss-scale",
        str(training.pitch_loss_scale),
        "--attn-loss-scale",
        str(training.attn_loss_scale),
        "--text-cleaners",
        *tokenizer.text_cleaners,
        "--symbol-set",
        tokenizer.symbol_set,
        "--p-arpabet",
        str(tokenizer.p_arpabet),
        "--training-files",
        *[str(f) for f in train_files],
        "--validation-files",
        *[str(f) for f in val_files],
        "--n-speakers",
        str(training.n_speakers),
        "--seed",
        str(training.seed),
    ]

    if tokenizer.style_tags:
        args.extend(["--style-tags", *tokenizer.style_tags])
    if tokenizer.strip_style_from_text:
        args.append("--strip-style-from-text")
    if training.amp:
        args.append("--amp")
    if training.cudnn_benchmark:
        args.append("--cudnn-benchmark")
    if use_cuda:
        args.append("--cuda")

    return args


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train FastPitch with locked style-tag tokenizer settings",
        allow_abbrev=False,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to FastPitch config JSON")
    parser.add_argument("--metadata", help="metadata.csv or filelist to split into train/val")
    parser.add_argument("--train-filelist", nargs="*", help="Existing training filelists (skip --metadata)")
    parser.add_argument("--val-filelist", nargs="*", help="Existing validation filelists")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split when using --metadata")
    parser.add_argument("--output", required=True, help="Checkpoint output directory")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA training")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, help="Override batch size from config")

    args = parser.parse_args(argv)
    project_config = load_fastpitch_config(Path(args.config))

    if args.metadata:
        train_files, val_files = _split_metadata(
            Path(args.metadata), Path(args.output) / "filelists", args.val_ratio, project_config.training.seed
        )
    elif args.train_filelist and args.val_filelist:
        train_files = _resolve_filelists(args.train_filelist)
        val_files = _resolve_filelists(args.val_filelist)
    else:
        raise ValueError("Provide either --metadata or both --train-filelist and --val-filelist")

    cli_args = _build_training_cli(
        config=project_config,
        output_dir=Path(args.output),
        train_files=train_files,
        val_files=val_files,
        use_cuda=args.cuda,
        override_epochs=args.epochs,
        override_batch=args.batch_size,
    )

    print(project_config.describe())
    print(f"Using training files: {train_files}")
    print(f"Using validation files: {val_files}")
    print(f"Launching FastPitch training with args: {' '.join(cli_args)}")

    fastpitch_train.main(cli_args)


if __name__ == "__main__":
    main()
