# New-task rebuttal figure

Final asset layout:

- `overview_plus_rollouts_preview.png`: locked plug/fork overview and rollout montage.
- `success_over_time.png` / `.pdf`: full-task success rate during finetuning.
- `new_tasks_with_training_curve.png` / `.pdf`: montage on the left, training plot on the right.
- `success_over_time_curves.json`: cached, smoothed TensorBoard curve data and provenance.

The raw training metric is `all_goals_hit_ratio`, the fraction of episodes that
complete both the pre-insert and final goals. Because the training distribution
contains physically impossible initializations, the plotted curves are scaled
to terminal feasible-initialization success rates of 90% for plug and 98% for
fork. Curves use a centered ten-minute rolling mean and contain only observed
training time; they are not extended after a run ends. The JSON cache retains
the raw, unnormalized curves.

Regenerate with:

```bash
.venv/bin/python plot_figures/rebuttal/final_plots/05_new_tasks/success_over_time.py --refresh
```
