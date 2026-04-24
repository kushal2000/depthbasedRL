from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym", required=True)
    parser.add_argument("--lab", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gym = np.load(args.gym)
    lab = np.load(args.lab)
    keys = sorted(set(gym.files) & set(lab.files))
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        g = gym[key]
        l = lab[key]
        if g.shape != l.shape:
            summary[key] = {"shape_mismatch": 1.0}
            continue
        diff = np.abs(g - l)
        summary[key] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    main()
