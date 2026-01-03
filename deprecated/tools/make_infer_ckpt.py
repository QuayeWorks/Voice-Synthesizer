# make_infer_ckpt.py
import os
import sys
import torch

# adjust these to your paths
ROOT = r"C:\Users\Emmanuel\Desktop\Projects\QuayeWorks\VoiceSynthesizer"
tacotron_dir = os.path.join(ROOT, "third_party", "tacotron2")

# this is your big 322MB training checkpoint
src_ckpt_path = os.path.join(ROOT, "datasets", "output", "checkpoint_9000")

# this is the smaller, inference-friendly file we will create
dst_ckpt_path = os.path.join(ROOT, "checkpoints", "tacotron2_infer_adapted.pt")

# 1) import tacotron2 stuff from NVIDIA repo
if tacotron_dir not in sys.path:
    sys.path.insert(0, tacotron_dir)

from hparams import create_hparams
from model import Tacotron2

print(f"[export] loading training checkpoint: {src_ckpt_path}")
ckpt = torch.load(src_ckpt_path, map_location="cpu")

# 2) build hparams
# start from the repo defaults
hparams = create_hparams()

# if the training checkpoint already had a config (like your working 'latest' did),
# we should apply it here so we rebuild THE SAME MODEL.
ckpt_cfg = None
if isinstance(ckpt, dict) and "config" in ckpt:
    ckpt_cfg = ckpt["config"]
    print("[export] found 'config' in source checkpoint, overlaying onto hparams")
    for k, v in ckpt_cfg.items():
        if hasattr(hparams, k):
            setattr(hparams, k, v)

# you were force-setting these before — keep them if they match your training
hparams.sampling_rate   = 22050
hparams.filter_length   = 1024
hparams.hop_length      = 256
hparams.win_length      = 1024
hparams.n_mel_channels  = 80
hparams.text_cleaners   = ['english_cleaners']

# 3) build a fresh model with these hparams
model = Tacotron2(hparams)

# 4) get the weights from the training ckpt
# some checkpoints are {"state_dict": ..., "optimizer": ...}
# others are just a raw state dict
src_state = ckpt.get("state_dict", ckpt)

# 5) strip "module." if present
clean_state = {}
for k, v in src_state.items():
    if k.startswith("module."):
        clean_state[k[len("module."):]] = v
    else:
        clean_state[k] = v

# 6) load into the fresh model
missing, unexpected = model.load_state_dict(clean_state, strict=False)
if missing:
    print("[export] WARNING: missing keys when loading:", missing)
if unexpected:
    print("[export] WARNING: unexpected keys when loading:", unexpected)

model.eval()

# 7) turn hparams into a plain dict so synthesize.py can re-create it later
# (vars(hparams) works for the NVIDIA repo hparams)
cfg_dict = {k: v for k, v in vars(hparams).items() if not k.startswith("_")}

# 8) save in the same style as your working checkpoint:
#    { "config": {...}, "state_dict": {...} }
to_save = {
    "config": cfg_dict,
    "state_dict": model.state_dict(),
}

torch.save(to_save, dst_ckpt_path)
print(f"[export] wrote normalized inference checkpoint to {dst_ckpt_path}")
