# Offline teacher eval

Success rate for a state-obs SAPG teacher, at the checkpoint's own training config.

## The 4 checkpoints

`RUNDIR` below; the checkpoint is `$RUNDIR/*/last/model.pth` and the config is
`$RUNDIR/.hydra/overrides.yaml`. All paths relative to the repo root.

| Task | RUNDIR | Problem |
|---|---|---|
| peg | `train_dir/teachers_final/play2win_peg_insertion/lpeg_tol0p5mm_finetune_rgf0_dr_2026-05-15_21-56-40` | `Lpeg_matchedmass.tol0p5mm` |
| beam part 0 | `train_dir/fig4/beam_3x_teachers/beam_3x_part_0_finetune_rgf0_dr_wrench_2026-05-25_05-40-24` | `fabrica.beam_3x.part_0_matchedmass_sdf_hybrid` |
| beam part 2 | `train_dir/fig4/beam_3x_teachers/beam_3x_part_2_finetune_rgf0_dr_wrench_2026-05-25_05-40-24` | `fabrica.beam_3x.part_2_matchedmass_sdf_hybrid` |
| furniture_bench | `train_dir/fig4/long_leg_screwing_teachers/furniture_bench_leg4_200mm_finetune_rgf10_dr_wrench_2026-05-25_22-16-15` | `furniture_bench.one_leg_leg4_200mm_matchedmass_sdf_hybrid_super_dense` |

Per-task flags on top of the standard command:

- **beam ×2, furniture_bench** are `_dr_wrench` teachers (trained at
  `force_scale=20.0`, `torque_scale=2.0`). To eval without wrench:
  `--override env.domain_randomization.force_scale=0.0 --override env.domain_randomization.torque_scale=0.0`
- **furniture_bench** trained at `rgf=0.1`; add
  `--override env.peg_in_hole.random_goal_fraction=0.0` so every env runs the
  insertion task rather than the random-goal co-training mix.
- **peg** has no wrench-trained teacher — this is the plain `_dr` policy, already
  `force_scale=0.0`, so no override needed.

## Run

```bash
RUN=train_dir/teachers_final/play2win_peg_insertion/lpeg_tol0p5mm_finetune_rgf0_dr_2026-05-15_21-56-40
CK=$RUN/0_lpeg_tol0p5mm_finetune_rgf0_dr_2026-05-15_21-56-40/last/model.pth

OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -u \
  peg_in_hole_dynamic/offline_eval_teacher_robustness.py \
  --teacher-checkpoint "$CK" \
  --train-overrides $RUN/.hydra/overrides.yaml \
  --exact-train --num-envs 512 --seed 42 \
  --output-json /tmp/eval.json
```

`--exact-train` replays every `env.*` line from the checkpoint's `overrides.yaml`
(problem, goal_mode, tolerances, retract, hole yaw, full DR block). **Always use
it** — the script's own defaults differ from training and silently give wrong
numbers. Takes ~2 min Kit boot + ~80 s.

**Kit hangs on teardown.** Kill the process once `=> wrote <json>` appears:

```bash
pkill -9 -f offline_eval_teacher_robustness.py
```

## Output

```
all envs   (n= 510): insertion=0.8725 (445)  retract=0.8725 (445)
no early drop (n= 452): insertion=0.9845 (445)  retract=0.9845 (445)
terminations: {'max_successes': 445, 'fall': 61, ...}  dropped_early=58/61
```

- **`no early drop`** is the number to report. Excludes envs whose object fell
  before step 100 (`--early-drop-steps`) — unstable initial placement, not a
  policy failure. Reset noise is ±10 cm in x/y, so this is common.
- **`retract`** is the env's real success flag when `enable_retract=True`:
  inserted, hand withdrawn 10 cm, peg still within 7.5 mm of goal.
  **`insertion`** is the weaker "peg went in" criterion.
- One episode per env, so each env is one unweighted sample.
- **`unfinished` must be ~0.** If it isn't, `--max-steps-per-episode` was too
  short and the censored envs are silently dropped from the denominator,
  inflating the rate. Scale the budget with the problem's goal count: the env
  resets `episode_length_buf` on *every* subgoal success
  (`peg_in_hole_env.py:604`), so a 10-goal problem can legitimately run far past
  the 600-step default. peg/beam (`goals=2`) are fine at 600;
  furniture_bench (`goals=10`) needs `--max-steps-per-episode 3000`.

## Extra overrides

`--override KEY=VALUE` (repeatable) applies after `--train-overrides`:

```bash
  --override env.domain_randomization.force_scale=0.0 \
  --override env.domain_randomization.torque_scale=0.0
```

Only needed for checkpoints trained *with* wrench DR (`fig4/*_dr_wrench*`).
Checkpoints without it already carry `force_scale=0.0` in their training config.

Turning wrench **on** needs more than the scale: `force_prob_range` /
`torque_prob_range` default to `(0.001, 0.1)`, so perturbations may almost never
fire. Set them too, or the disturbance is effectively a no-op.

## Gotchas

- Success must come from `env._termination_reasons`, not `env._successes` —
  `DirectRLEnv.step()` resets terminated envs before returning, so `_successes`
  is zeroed before you can read it. Reading it makes the terminal goal invisible.
- Don't filter on drop-terminations generally: a successful episode can't end in
  a drop, so it removes only failures and can only inflate the rate. Filter on
  *timing* (`--early-drop-steps`) instead.
- `offline_eval_5way.py`, `offline_eval_robustness.py`, `offline_eval_table_dr.py`
  and `offline_eval_one_backend.py` still use the broken `_successes` metric.
