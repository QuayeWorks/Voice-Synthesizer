# inspect_ckpt.py
import argparse
import torch
from pprint import pprint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to tacotron2 .pt / checkpoint_xxx file")
    args = ap.parse_args()

    print(f"[info] loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # 1) show top-level keys
    print("\n[info] top-level keys in checkpoint:")
    top_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
    for k in top_keys:
        print("  -", k)

    # --- detect style -------------------------------------------------------
    # style A: trainer-style -> has "state_dict" (and often "epoch", "config")
    # style B: bare state dict -> top-level keys look like real layer names
    style = "unknown"

    state_dict = None

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        style = "trainer-style (wrapped)"
        state_dict = ckpt["state_dict"]
    else:
        # maybe it's a bare state dict
        # heuristic: keys look like module params, e.g. 'encoder.lstm.weight_ih_l0'
        if isinstance(ckpt, dict):
            # if MOST of the keys look like pytorch params, call it bare
            paramish = 0
            for k in ckpt.keys():
                if "." in k or "weight" in k or "bias" in k:
                    paramish += 1
            if paramish >= max(1, len(ckpt) // 2):
                style = "bare state_dict"
                state_dict = ckpt  # it's the weights
            else:
                style = "dict (nonstandard)"
        else:
            style = f"unsupported type: {type(ckpt)}"

    print(f"\n[info] detected checkpoint style: {style}")

    # 2) try to show hparams / config if present
    hp = None
    if isinstance(ckpt, dict):
        hp = ckpt.get("hparams", None)
        cfg = ckpt.get("config", None)
    else:
        hp = None
        cfg = None

    if hp is not None:
        print("\n[info] 'hparams' found:")
        if isinstance(hp, dict):
            for k, v in hp.items():
                print(f"  {k}: {v}")
        else:
            for k, v in vars(hp).items():
                print(f"  {k}: {v}")
    else:
        print("\n[info] no 'hparams' found in this checkpoint.")

    if cfg is not None:
        print("\n[info] 'config' found:")
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                print(f"  {k}: {v}")
        else:
            for k, v in vars(cfg).items():
                print(f"  {k}: {v}")
    else:
        print("\n[info] no 'config' found in this checkpoint.")

    # 3) show how big the state_dict is
    if state_dict is None and isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"].get("state_dict", None)

    if state_dict is not None:
        keys = list(state_dict.keys())
        print(f"\n[info] state_dict has {len(keys)} params. showing first {min(len(keys), 120)} keys:")
        for i, k in enumerate(keys):
            print("   ", k)
            if i >= 119:
                print("   ...")
                break
    else:
        print("\n[info] no state_dict found (maybe this is a completely custom object?).")

    # 4) small hint for user based on style
    if style == "bare state_dict":
        print("\n[hint] this looks like a raw model.state_dict().")
        print("       to use it, you must build your Tacotron2 with the SAME hparams you trained with,")
        print("       then do model.load_state_dict(torch.load(...)).")
    elif style == "trainer-style (wrapped)":
        print("\n[hint] this looks like a trainer/DP-style checkpoint with a wrapped state_dict.")
        print("       you will probably need to strip 'module.' prefixes before loading into a non-DP model.")

if __name__ == "__main__":
    main()
