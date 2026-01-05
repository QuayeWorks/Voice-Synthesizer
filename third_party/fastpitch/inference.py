# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

"""FastPitch inference (text -> mel) ONLY.

This file was patched to remove all Tacotron2/WaveGlow-specific paths so it can be
used cleanly in a modular pipeline:

    Text -> FastPitch -> Mel (.npy) -> HiFi-GAN -> WAV

Use (CLI):
    python third_party\\fastpitch\\inference.py -i phrases\\my_lines.txt -o inference\\mels --cuda \
        --fastpitch <FASTPITCH_CKPT> --save-mels

NOTE (Library/GUI use):
- This module can be imported and used by other Python code (e.g., a PyQt GUI).
- In that case, we must NOT require CLI-only flags at import-time or when
  helper functions call argparse. Required CLI flags are enforced only in main().
"""

import argparse
import time
import warnings
from pathlib import Path

from tqdm import tqdm

import torch
import numpy as np
from scipy.stats import norm
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from . import models
from .common.tb_dllogger import (
    init_inference_metadata,
    stdout_metric_format,
    unique_log_fpath,
)
from .common.text import cmudict
from .common.text.symbols import STYLE_TAGS, get_symbols
from .common.text.text_processing import TextProcessing
from .pitch_transform import pitch_transform_custom


def parse_args(parser):
    """Define commandline arguments.

    IMPORTANT:
    - We DO NOT mark -i/--input and --fastpitch as required here, so that this
      module can be imported and used as a library (e.g., by a GUI) without
      argparse aborting.
    - CLI requirements are enforced in main().
    """
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,   # was required=True
        help="Full path to the input text (phrases separated by newlines). "
             "Also supports .tsv with a header row.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output folder (used when --save-mels is set).",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Path to a DLLogger log file")
    parser.add_argument("--save-mels", action="store_true", help="Save predicted mels as .npy files")
    parser.add_argument("--cuda", action="store_true", help="Run inference on a GPU using CUDA")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Enable cudnn benchmark mode")
    parser.add_argument(
        "--fastpitch",
        type=str,
        default=None,   # was required=True
        help="Full path to the FastPitch checkpoint file (use SKIP to load ground-truth mels from TSV).",
    )
    parser.add_argument("--amp", action="store_true", help="Inference with AMP")
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup iterations before measuring performance")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat inference for benchmarking")
    parser.add_argument("--torchscript", action="store_true", help="Apply TorchScript")
    parser.add_argument("--ema", action="store_true", help="Use EMA averaged model (if saved in checkpoints)")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to dataset root (only needed if you load ground truth mels from a TSV).",
    )
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID for a multi-speaker model")

    parser.add_argument("--p-arpabet", type=float, default=0.0, help="Probability of converting words to ARPAbet")
    parser.add_argument("--heteronyms-path", type=str, default="cmudict/heteronyms", help="")
    parser.add_argument("--cmudict-path", type=str, default="cmudict/cmudict-0.7b", help="")
    parser.add_argument(
        "--strip-style-from-text",
        action="store_true",
        help="Remove leading style tag from text sequence but return style id separately (future conditioning)",
    )
    parser.add_argument(
        "--style-tags",
        nargs="*",
        default=STYLE_TAGS,
        help="Optional override list of supported style tags",
    )

    transform = parser.add_argument_group("transform")
    transform.add_argument("--pace", type=float, default=1.0, help="Adjust the pace of speech")
    transform.add_argument("--pitch-transform-flatten", action="store_true", help="Flatten the pitch")
    transform.add_argument("--pitch-transform-invert", action="store_true", help="Invert the pitch wrt mean value")
    transform.add_argument(
        "--pitch-transform-amplify",
        type=float,
        default=1.0,
        help="Amplify pitch variability, typical values are in the range (1.0, 3.0).",
    )
    transform.add_argument("--pitch-transform-shift", type=float, default=0.0, help="Raise/lower the pitch by <hz>")
    transform.add_argument("--pitch-transform-custom", action="store_true", help="Apply the transform from pitch_transform.py")

    text_processing = parser.add_argument_group("Text processing parameters")
    text_processing.add_argument(
        "--text-cleaners",
        nargs="*",
        default=["english_cleaners_v2"],
        type=str,
        help="Type of text cleaners for input text",
    )
    text_processing.add_argument("--symbol-set", type=str, default="english_basic", help="Define symbol set for input text")

    cond = parser.add_argument_group("conditioning on additional attributes")
    cond.add_argument("--n-speakers", type=int, default=1, help="Number of speakers in the model.")

    return parser


def load_model_from_ckpt(checkpoint_path, ema, model):
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    status = ""

    if "state_dict" in checkpoint_data:
        sd = checkpoint_data["state_dict"]
        if ema and "ema_state_dict" in checkpoint_data:
            sd = checkpoint_data["ema_state_dict"]
            status += " (EMA)"
        elif ema and "ema_state_dict" not in checkpoint_data:
            print(f"WARNING: EMA weights missing for {checkpoint_path}")

        if any(key.startswith("module.") for key in sd):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        status += " " + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data["model"]

    print(f"Loaded {checkpoint_path}{status}")
    return model


def load_and_setup_model(
    model_name,
    parser,
    checkpoint,
    amp,
    device,
    unk_args=None,
    forward_is_infer=False,
    ema=True,
    jitable=False,
):
    """Library-safe model loader.

    IMPORTANT FIX:
    - Do NOT parse real sys.argv here. Some embedding apps (GUIs) will import
      this module and call load_and_setup_model, and sys.argv may not include
      FastPitch CLI flags.
    - We parse model args from an empty argv so argparse never aborts.
    """
    if unk_args is None:
        unk_args = []

    model_parser = models.parse_model_args(model_name, parser, add_help=False)

    # FIX: do not read sys.argv
    model_args, model_unk_args = model_parser.parse_known_args([])

    # Keep only the unknown args that the model parser also considers unknown
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(
        model_name,
        model_config,
        device,
        forward_is_infer=forward_is_infer,
        jitable=jitable,
    )

    if checkpoint is not None and checkpoint != "SKIP":
        model = load_model_from_ckpt(checkpoint, ema, model)

    if amp:
        model.half()
    model.eval()
    return model.to(device)


def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding="utf-8")]
    if fpath.endswith(".tsv"):
        columns = lines[0].split("\t")
        fields = list(zip(*[t.split("\t") for t in lines[1:]]))
    else:
        columns = ["text"]
        fields = [lines]
    return {c: f for c, f in zip(columns, fields)}


def prepare_input_sequence(
    fields,
    device,
    symbol_set,
    text_cleaners,
    batch_size=128,
    dataset=None,
    load_mels=False,
    load_pitch=False,
    p_arpabet=0.0,
    include_style_tokens=True,
    style_tags=None,
    strip_style_from_text=False,
):
    tp = TextProcessing(
        symbol_set,
        text_cleaners,
        p_arpabet=p_arpabet,
        include_style_tokens=include_style_tokens,
        style_tags=style_tags,
        strip_style_from_text=strip_style_from_text,
    )

    fields["text"] = [torch.LongTensor(tp.encode_text(text)) for text in fields["text"]]
    order = np.argsort([-t.size(0) for t in fields["text"]])

    fields["text"] = [fields["text"][i] for i in order]
    fields["text_lens"] = torch.LongTensor([t.size(0) for t in fields["text"]])

    for t in fields["text"]:
        print(tp.sequence_to_text(t.numpy()))

    if load_mels:
        assert "mel" in fields, "TSV must contain a 'mel' column when loading ground-truth mels."
        assert dataset is not None, "--dataset-path is required when loading ground-truth mels."
        fields["mel"] = [torch.load(Path(dataset, fields["mel"][i])).t() for i in order]
        fields["mel_lens"] = torch.LongTensor([t.size(0) for t in fields["mel"]])

    if load_pitch:
        assert "pitch" in fields, "TSV must contain a 'pitch' column when loading pitch."
        assert dataset is not None, "--dataset-path is required when loading pitch."
        fields["pitch"] = [torch.load(Path(dataset, fields["pitch"][i])) for i in order]
        fields["pitch_lens"] = torch.LongTensor([t.size(0) for t in fields["pitch"]])

    if "output" in fields:
        fields["output"] = [fields["output"][i] for i in order]

    # Cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b : b + batch_size] for f, values in fields.items()}
        for f in batch:
            if f == "text":
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == "mel" and load_mels:
                batch[f] = pad_sequence(batch[f], batch_first=True).permute(0, 2, 1)
            elif f == "pitch" and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if isinstance(batch[f], torch.Tensor):
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches


def build_pitch_transformation(args):
    if args.pitch_transform_custom:
        def custom_(pitch, pitch_lens, mean, std):
            return (pitch_transform_custom(pitch * std + mean, pitch_lens) - mean) / std
        return custom_

    fun = "pitch"
    if args.pitch_transform_flatten:
        fun = f"({fun}) * 0.0"
    if args.pitch_transform_invert:
        fun = f"({fun}) * -1.0"
    if args.pitch_transform_amplify:
        ampl = args.pitch_transform_amplify
        fun = f"({fun}) * {ampl}"
    if args.pitch_transform_shift != 0.0:
        hz = args.pitch_transform_shift
        fun = f"({fun}) + {hz} / std"
    return eval(f"lambda pitch, pitch_lens, mean, std: {fun}")


class MeasureTime(list):
    def __init__(self, *args, cuda=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)


def main():
    """Launches FastPitch inference (text -> mel) CLI."""
    parser = argparse.ArgumentParser(description="PyTorch FastPitch Inference (mel-only)", allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    # ENFORCE REQUIRED ARGS ONLY FOR CLI
    if args.input is None or args.fastpitch is None:
        parser.error("the following arguments are required: -i/--input, --fastpitch")

    symbol_table = get_symbols(
        args.symbol_set, include_style_tokens=True, style_tags=args.style_tags
    )
    print(
        "Tokenizer: "
        f"symbol_set={args.symbol_set} (n_symbols={len(symbol_table)}) | "
        f"text_cleaners={', '.join(args.text_cleaners)} | "
        f"style_tags={len(args.style_tags)} locked tags"
    )

    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, keep_ambiguous=True)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device = torch.device("cuda" if args.cuda else "cpu")

    # Output folder is required when saving mels
    if args.save_mels:
        if args.output is None:
            raise ValueError("When using --save-mels you must provide -o/--output.")
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # Logging (if output is None, still allow logging to cwd)
    log_base = Path(args.output) if args.output is not None else Path(".")
    log_fpath = args.log_file or str(Path(log_base, "nvlog_infer.json"))
    log_fpath = unique_log_fpath(log_fpath)
    DLLogger.init(
        backends=[
            JSONStreamBackend(Verbosity.DEFAULT, log_fpath),
            StdOutBackend(Verbosity.VERBOSE, metric_format=stdout_metric_format),
        ]
    )
    init_inference_metadata()
    [DLLogger.log("PARAMETER", {k: v}) for k, v in vars(args).items()]

    if args.fastpitch != "SKIP":
        generator = load_and_setup_model(
            "FastPitch",
            parser,
            args.fastpitch,
            args.amp,
            device,
            unk_args=unk_args,
            forward_is_infer=True,
            ema=args.ema,
            jitable=args.torchscript,
        )
        if args.torchscript:
            generator = torch.jit.script(generator)
    else:
        generator = None

    if len(unk_args) > 0:
        raise ValueError(f"Invalid options {unk_args}")

    fields = load_fields(args.input)
    batches = prepare_input_sequence(
        fields,
        device,
        args.symbol_set,
        args.text_cleaners,
        args.batch_size,
        args.dataset_path,
        load_mels=(generator is None),
        p_arpabet=args.p_arpabet,
        include_style_tokens=True,
        style_tags=args.style_tags,
        strip_style_from_text=args.strip_style_from_text,
    )

    gen_measures = MeasureTime(cuda=args.cuda)

    gen_kw = {
        "pace": args.pace,
        "speaker": args.speaker,
        "pitch_tgt": None,
        "pitch_transform": build_pitch_transformation(args),
    }

    if args.torchscript:
        gen_kw.pop("pitch_transform")
        print("NOTE: Pitch transforms are disabled with TorchScript")

    all_letters = 0
    all_frames = 0

    reps = args.repeats
    log_enabled = reps == 1
    log = lambda s, d: DLLogger.log(step=s, data=d) if log_enabled else None

    # Warmup
    for _ in tqdm(range(args.warmup_steps), "Warmup"):
        with torch.no_grad():
            if generator is not None:
                b = batches[0]
                _mel, *_ = generator(b["text"], **gen_kw)

    for rep in (tqdm(range(reps), "Inference") if reps > 1 else range(reps)):
        for b in batches:
            if generator is None:
                log(rep, {"Synthesizing from ground truth mels"})
                mel, mel_lens = b["mel"], b["mel_lens"]
            else:
                with torch.no_grad(), gen_measures:
                    mel, mel_lens, *_ = generator(b["text"], **gen_kw)

                gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
                all_letters += b["text_lens"].sum().item()
                all_frames += mel.size(0) * mel.size(2)
                log(rep, {"fastpitch_frames/s": gen_infer_perf})
                log(rep, {"fastpitch_latency": gen_measures[-1]})

                if args.save_mels:
                    if "global_mel_idx" not in locals():
                        global_mel_idx = 0

                    for i, mel_ in enumerate(mel):
                        m = mel_[:, : mel_lens[i].item()].permute(1, 0)

                        if "output" in b:
                            fname = b["output"][i]
                            stem = Path(fname).stem
                        else:
                            stem = f"mel_{global_mel_idx:05d}"

                        mel_path = Path(args.output, stem + ".npy")
                        np.save(mel_path, m.cpu().numpy())

                        global_mel_idx += 1

    # Summary stats (FastPitch only)
    if generator is not None and len(gen_measures) > 0:
        gm = np.sort(np.asarray(gen_measures))
        log((), {"avg_fastpitch_letters/s": all_letters / gm.sum()})
        log((), {"avg_fastpitch_frames/s": all_frames / gm.sum()})
        log((), {"avg_fastpitch_latency": gm.mean()})
        log((), {"90%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std()})
        log((), {"95%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std()})
        log((), {"99%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std()})

    DLLogger.flush()


if __name__ == "__main__":
    main()
