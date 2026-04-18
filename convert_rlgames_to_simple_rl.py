"""Convert a pretrained rl_games (SAPG) checkpoint to simple_rl format.

The two formats are architecturally identical for SAPG M=6 conditioning_dim=32,
with only one key rename required:
    a2c_network.extra_params  →  a2c_network.conditioning

All other keys (MLP weights, LSTM, mu, value, layer norm, running stats) are
unchanged in name and shape.

Usage
-----
python convert_rlgames_to_simple_rl.py \
    --input  pretrained_policy/model.pth \
    --output pretrained_policy/model_simple_rl.pth

Then evaluate with:
python dextoolbench/eval.py \
    --object-category hammer --object-name claw_hammer --task-name swing_down \
    --config-path pretrained_policy/config_simple_rl.yaml \
    --checkpoint-path pretrained_policy/model_simple_rl.pth \
    --use-simple-rl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert(input_path: Path, output_path: Path) -> None:
    print(f"Loading rl_games checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")

    rank_data = ckpt[0]
    state_dict = rank_data["model"]

    print("Keys in rl_games checkpoint:")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {tuple(v.shape)}")

    # ── The one rename ───────────────────────────────────────────────────────
    if "a2c_network.extra_params" in state_dict:
        state_dict["a2c_network.conditioning"] = state_dict.pop("a2c_network.extra_params")
        print("\nRenamed: a2c_network.extra_params → a2c_network.conditioning")
    else:
        print("\nWARNING: 'a2c_network.extra_params' not found — checkpoint may already be converted.")

    print("\nKeys in converted checkpoint:")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {tuple(v.shape)}")

    rank_data["model"] = state_dict
    ckpt[0] = rank_data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"\nSaved converted checkpoint to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  type=Path, required=True, help="Path to rl_games .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output path for simple_rl .pth checkpoint")
    args = parser.parse_args()

    assert args.input.exists(), f"Input not found: {args.input}"
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
