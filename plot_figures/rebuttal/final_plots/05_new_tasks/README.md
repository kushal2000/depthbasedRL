# New-task rebuttal figure

Final asset layout:

- `overview_plus_rollouts_preview.png`: two-row plug/fork overview and rollout montage with row headers.
- `success_over_time.png` / `.pdf`: full-task success rate during finetuning.
- `new_tasks_with_training_curve.png` / `.pdf`: compact, approximately 2.9:1
  composite with the montage directly adjoining the training plot.
- `success_over_time_curves.json`: cached, smoothed TensorBoard curve data and provenance.

The raw training metric is `all_goals_hit_ratio`, the fraction of episodes that
complete both the pre-insert and final goals. Because the training distribution
contains physically impossible initializations, the plotted curves are scaled
to the latest early-drop-filtered offline insertion rates: 98.62% for Plug at
0.5 mm and 98.40% for Fork at 1 mm. Curves use a centered ten-minute rolling
mean and contain only observed training time; they are not extended after a run
ends. The JSON cache retains the raw, unnormalized curves.

Regenerate with:

```bash
.venv/bin/python plot_figures/rebuttal/final_plots/05_new_tasks/success_over_time.py --refresh
```
