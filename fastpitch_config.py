"""Shared FastPitch tokenizer/training configuration helpers.

These helpers keep the GUI, inference CLI, and training entrypoints
in sync so the tokenizer + style tag inventory cannot silently drift.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from third_party.fastpitch.common.text.symbols import STYLE_TAGS, get_symbols

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "fastpitch_expressive.json"


@dataclass
class TokenizerConfig:
    symbol_set: str
    text_cleaners: List[str]
    include_style_tokens: bool
    style_tags: List[str]
    strip_style_from_text: bool
    p_arpabet: float
    expected_n_symbols: int | None

    @classmethod
    def from_dict(cls, data: dict) -> "TokenizerConfig":
        return cls(
            symbol_set=data["symbol_set"],
            text_cleaners=list(data.get("text_cleaners", [])),
            include_style_tokens=bool(data.get("include_style_tokens", True)),
            style_tags=list(data.get("style_tags", STYLE_TAGS)),
            strip_style_from_text=bool(data.get("strip_style_from_text", False)),
            p_arpabet=float(data.get("p_arpabet", 0.0)),
            expected_n_symbols=data.get("n_symbols"),
        )

    def symbol_inventory(self) -> List[str]:
        return get_symbols(
            self.symbol_set,
            include_style_tokens=self.include_style_tokens,
            style_tags=self.style_tags,
            extra_symbols=None,
        )

    def n_symbols(self) -> int:
        return len(self.symbol_inventory())

    def validate(self) -> None:
        symbols = self.symbol_inventory()
        missing_tags = [tag for tag in self.style_tags if tag not in symbols]
        if missing_tags:
            raise ValueError(f"Style tags missing from symbol table: {missing_tags}")

        if self.include_style_tokens and set(self.style_tags) != set(STYLE_TAGS):
            raise ValueError("Config style_tags must match the locked 10-style inventory.")

        if self.expected_n_symbols is not None and self.expected_n_symbols != len(symbols):
            raise ValueError(
                "Config n_symbols does not match derived symbol count "
                f"({self.expected_n_symbols} != {len(symbols)})."
            )

    def summary_lines(self) -> List[str]:
        symbol_line = f"symbol_set={self.symbol_set} (n_symbols={self.n_symbols()})"
        cleaners_line = f"text_cleaners={', '.join(self.text_cleaners)}"
        style_mode = "strip" if self.strip_style_from_text else "keep"
        style_line = (
            f"style_tags={len(self.style_tags)} locked tags ({style_mode} tag token in text)"
        )
        return [symbol_line, cleaners_line, style_line]


@dataclass
class TrainingConfig:
    dataset_path: Path
    batch_size: int
    learning_rate: float
    epochs: int
    epochs_per_checkpoint: int
    grad_accumulation: int
    amp: bool
    cudnn_benchmark: bool
    warmup_steps: int
    dur_loss_scale: float
    pitch_loss_scale: float
    attn_loss_scale: float
    n_speakers: int
    seed: int

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        return cls(
            dataset_path=Path(data.get("dataset_path", "datasets")),
            batch_size=int(data["batch_size"]),
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            epochs_per_checkpoint=int(data.get("epochs_per_checkpoint", 10)),
            grad_accumulation=int(data.get("grad_accumulation", 1)),
            amp=bool(data.get("amp", False)),
            cudnn_benchmark=bool(data.get("cudnn_benchmark", False)),
            warmup_steps=int(data.get("warmup_steps", 1000)),
            dur_loss_scale=float(data.get("dur_loss_scale", 1.0)),
            pitch_loss_scale=float(data.get("pitch_loss_scale", 1.0)),
            attn_loss_scale=float(data.get("attn_loss_scale", 1.0)),
            n_speakers=int(data.get("n_speakers", 1)),
            seed=int(data.get("seed", 1234)),
        )


@dataclass
class FastPitchProjectConfig:
    tokenizer: TokenizerConfig
    training: TrainingConfig

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "FastPitchProjectConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = TokenizerConfig.from_dict(payload["tokenizer"])
        tokenizer.validate()
        training = TrainingConfig.from_dict(payload["training"])
        return cls(tokenizer=tokenizer, training=training)

    def describe(self) -> str:
        lines: Iterable[str] = ["FastPitch settings:"]
        lines = list(lines) + [f"  - {line}" for line in self.tokenizer.summary_lines()]
        lines.append(
            f"  - training: batch_size={self.training.batch_size}, lr={self.training.learning_rate}, "
            f"epochs={self.training.epochs}, grad_accumulation={self.training.grad_accumulation}"
        )
        lines.append(
            f"  - amp={self.training.amp}, cudnn_benchmark={self.training.cudnn_benchmark}, "
            f"warmup_steps={self.training.warmup_steps}"
        )
        return "\n".join(lines)


def load_fastpitch_config(path: Path | None = None) -> FastPitchProjectConfig:
    return FastPitchProjectConfig.load(path or DEFAULT_CONFIG_PATH)
