#!/usr/bin/env python3
"""Generate the fig-3 pretraining-backbone ablation finetuning `.sub` files.

Fig 3 holds the downstream finetuning recipe (task, SAPG agent, full DR) fixed
and varies ONLY the pretrained checkpoint we finetune from (+ seed). We sweep:

    9 checkpoints (1 Play2Win reference + 8 ablated, across 4 axes)
  x 4 flagship tasks (peg, beam_part0, beam_part2, furniture_bench)
  x 3 seeds (0, 1, 2)
  = 108 standalone, sbatch-able .sub files

Each file is rendered from `_template.sub` by replacing @@TOKEN@@ markers, and
written to a neat tree that mirrors the train-time output layout:

    runs/<task>/<axis>/<level>/seed<N>.sub      (ablated checkpoints)
    runs/<task>/play2win/seed<N>.sub            (shared reference)

The train-time HYDRA_RUN_DIR uses the same <task>/<axis>/<level>/seed<N> subpath
(under train_dir/fig3_ablations/), so checkpoints/logs/W&B land in the same
organization as the .sub tree.

Usage:
    python generate_subs.py            # write all 108 files
    python generate_subs.py --print    # dry-run: list jobs, verify checkpoints
"""

import argparse
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "_template.sub")
RUNS_DIR = os.path.join(HERE, "runs")
# Output tree the runs write into; reused runs are symlinked here so they sit
# alongside the freshly-launched seeds for plotting.
FIG3_OUT = "/share/portal/kk837/depthbasedRL/train_dir/fig3_ablations"

PRETRAIN_ROOT = "/share/portal/kk837/depthbasedRL/train_dir/pretraining_ablations"

WANDB_PROJECT = "fig3_ablations"
SEEDS = [0, 1, 2]


# ── The 9 checkpoints ────────────────────────────────────────────────────────
# `axis` groups them for plotting; "play2win" is the shared reference drawn on
# every axis panel and lives once per task (not duplicated per axis).
# `slug` is a short, filesystem/W&B-safe label used in run names.
def _ckpt(rel):
    return os.path.join(PRETRAIN_ROOT, rel)


CHECKPOINTS = [
    # Play2Win reference: 1000 obj, 6D pose, random goals, 1cm precision.
    {"axis": "play2win", "level": "play2win", "slug": "play2win",
     "path": _ckpt("pretrained_policy/model.pth")},

    # object_diversity
    {"axis": "object_diversity", "level": "obj10", "slug": "objdiv_obj10",
     "path": _ckpt("object_diversity/simtoolreal_pretrain_obj10_2026-05-17_22-19-44/0_simtoolreal_sapg/last/model.pth")},
    {"axis": "object_diversity", "level": "obj100", "slug": "objdiv_obj100",
     "path": _ckpt("object_diversity/simtoolreal_pretrain_obj100_2026-05-17_22-19-47/0_simtoolreal_sapg/last/model.pth")},

    # goal_precision
    {"axis": "goal_precision", "level": "prec5cm", "slug": "prec_5cm",
     "path": _ckpt("goal_precision/simtoolreal_pretrain_prec5cm_2026-05-24_01-07-03/0_simtoolreal_sapg/last/model.pth")},
    {"axis": "goal_precision", "level": "prec10cm", "slug": "prec_10cm",
     "path": _ckpt("goal_precision/simtoolreal_pretrain_prec10cm_2026-05-24_21-14-27/0_simtoolreal_sapg/last/model.pth")},

    # trajectory_diversity
    {"axis": "trajectory_diversity", "level": "traj10", "slug": "traj_10",
     "path": _ckpt("trajectory_diversity/simtoolreal_pretrain_traj10_2026-05-24_22-13-36/0_simtoolreal_sapg/last/model.pth")},
    {"axis": "trajectory_diversity", "level": "traj100", "slug": "traj_100",
     "path": _ckpt("trajectory_diversity/simtoolreal_pretrain_traj100_2026-05-24_01-58-22/0_simtoolreal_sapg/last/model.pth")},

    # training_objective (grasp has a different on-disk layout: runs/00_.../last)
    {"axis": "training_objective", "level": "rotation", "slug": "obj_rotation",
     "path": _ckpt("training_objective/simtoolreal_pretrain_rotation_2026-05-19_19-40-05/0_simtoolreal_sapg/last/model.pth")},
    {"axis": "training_objective", "level": "grasp", "slug": "obj_grasp",
     "path": _ckpt("training_objective/grasp_pretraining_2026-04-29_01-09-23/runs/00_grasp_pretraining_2026-04-29_01-09-23/last/model.pth")},
]


# ── The 4 flagship tasks (only these fields differ per task) ──────────────────
TASKS = {
    "peg": {
        "problem": "Lpeg_matchedmass.tol0p5mm",
        "rgf": "0.0", "insertion_tol": "0.010", "target_tol": "0.01",
    },
    "beam_part0": {
        "problem": "fabrica.beam_3x.part_0_matchedmass_sdf_hybrid",
        "rgf": "0.0", "insertion_tol": "0.010", "target_tol": "0.01",
    },
    "beam_part2": {
        "problem": "fabrica.beam_3x.part_2_matchedmass_sdf_hybrid",
        "rgf": "0.0", "insertion_tol": "0.010", "target_tol": "0.01",
    },
    "furniture_bench": {
        "problem": "furniture_bench.one_leg_leg4_200mm_matchedmass_sdf_hybrid_super_dense",
        "rgf": "0.1", "insertion_tol": "0.005", "target_tol": "0.002",
    },
}

# ── Reuse already-completed play2win _dr_wrench teacher runs ─────────────────
# These finetuned from the same play2win checkpoint with the same _dr_wrench
# recipe and downstream problem we use here, at seed 0. We reuse them as
# play2win/seed0 instead of relaunching: the generator writes NO .sub for these
# entries and instead symlinks them into the fig3_ablations output tree, so the
# reused run sits alongside the fresh seeds for plotting. peg has no wrench
# teacher, so it runs all seeds fresh.
TRAIN_DIR = "/share/portal/kk837/depthbasedRL/train_dir"
REUSE = {  # (task, seed) -> existing completed run dir to symlink as play2win
    ("beam_part0", 0): f"{TRAIN_DIR}/fig4/beam_3x_teachers/beam_3x_part_0_finetune_rgf0_dr_wrench_2026-05-25_05-40-24",
    ("beam_part2", 0): f"{TRAIN_DIR}/fig4/beam_3x_teachers/beam_3x_part_2_finetune_rgf0_dr_wrench_2026-05-25_05-40-24",
    ("furniture_bench", 0): f"{TRAIN_DIR}/fig4/long_leg_screwing_teachers/furniture_bench_leg4_200mm_finetune_rgf10_dr_wrench_2026-05-25_22-16-15",
}


def sub_subpath(task, ckpt):
    """Tree subpath shared by the .sub file and the train-time output dir."""
    if ckpt["axis"] == "play2win":
        return os.path.join(task, "play2win")
    return os.path.join(task, ckpt["axis"], ckpt["level"])


def render(task, ckpt, seed, template):
    cfg = TASKS[task]
    save_subpath = os.path.join(sub_subpath(task, ckpt), f"seed{seed}")
    experiment_tag = f"{task}__{ckpt['slug']}__seed{seed}"
    repl = {
        "@@PROBLEM@@": cfg["problem"],
        "@@RANDOM_GOAL_FRACTION@@": cfg["rgf"],
        "@@INSERTION_SUCCESS_TOLERANCE@@": cfg["insertion_tol"],
        "@@TARGET_SUCCESS_TOLERANCE@@": cfg["target_tol"],
        "@@CHECKPOINT@@": ckpt["path"],
        "@@SEED@@": str(seed),
        "@@WANDB_PROJECT@@": WANDB_PROJECT,
        "@@WANDB_GROUP@@": task,
        "@@EXPERIMENT_TAG@@": experiment_tag,
        "@@SAVE_SUBPATH@@": save_subpath,
    }
    out = template
    for k, v in repl.items():
        out = out.replace(k, v)
    if "@@" in out:
        leftover = sorted({tok for tok in out.split() if "@@" in tok})
        raise RuntimeError(f"Unsubstituted tokens in {experiment_tag}: {leftover}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="dry", action="store_true",
                    help="dry-run: list jobs and verify checkpoints, write nothing")
    args = ap.parse_args()

    with open(TEMPLATE) as f:
        template = f.read()

    # Verify checkpoints once up front.
    missing = [c for c in CHECKPOINTS if not os.path.isfile(c["path"])]
    missing_reuse = [(t, s, d) for (t, s), d in REUSE.items() if not os.path.isdir(d)]

    n_gen = 0
    n_reuse = 0
    for task in TASKS:
        for ckpt in CHECKPOINTS:
            for seed in SEEDS:
                # play2win seeds with an existing wrench teacher are reused, not
                # relaunched: symlink the run into the output tree, write no .sub.
                if ckpt["axis"] == "play2win" and (task, seed) in REUSE:
                    src = REUSE[(task, seed)]
                    link = os.path.join(FIG3_OUT, sub_subpath(task, ckpt), f"seed{seed}")
                    if args.dry:
                        print(f"  REUSE  fig3_ablations/{os.path.relpath(link, FIG3_OUT)} -> {src}")
                    else:
                        os.makedirs(os.path.dirname(link), exist_ok=True)
                        if os.path.islink(link) or os.path.exists(link):
                            os.remove(link)
                        os.symlink(src, link)
                    n_reuse += 1
                    continue

                rel = os.path.join(sub_subpath(task, ckpt), f"seed{seed}.sub")
                dest = os.path.join(RUNS_DIR, rel)
                if args.dry:
                    print(f"  runs/{rel}")
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "w") as f:
                        f.write(render(task, ckpt, seed, template))
                    os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR
                             | stat.S_IXGRP | stat.S_IXOTH)
                n_gen += 1

    total = n_gen + n_reuse
    print()
    print(f"{'DRY-RUN: would generate' if args.dry else 'Generated'} {n_gen} .sub files "
          f"+ {n_reuse} reused runs = {total} "
          f"({len(TASKS)} tasks x {len(CHECKPOINTS)} checkpoints x {len(SEEDS)} seeds)")
    print(f"  tasks:       {', '.join(TASKS)}")
    print(f"  checkpoints: {', '.join(c['slug'] for c in CHECKPOINTS)}")
    print(f"  seeds:       {SEEDS}")
    print(f"  reused:      {n_reuse} play2win seed-0 _dr_wrench teacher runs "
          f"(symlinked into fig3_ablations; no .sub, not resubmitted)")
    if not args.dry:
        print(f"  written under: {RUNS_DIR}")
    if missing_reuse:
        print("\n!! WARNING — reuse source dirs not found:")
        for t, s, d in missing_reuse:
            print(f"   [{t} seed{s}] {d}")
    if missing:
        print("\n!! WARNING — missing checkpoints (these runs will exit at submit time):")
        for c in missing:
            print(f"   [{c['slug']}] {c['path']}")
    else:
        print("\nAll 9 checkpoint paths verified to exist.")


if __name__ == "__main__":
    main()
