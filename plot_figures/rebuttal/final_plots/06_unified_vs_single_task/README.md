# Unified versus task-specific policies

`unified_vs_single_task.py` plots the early-drop-filtered insertion success
rate from the four fixed-fixture offline evaluations.
The source JSON paths live in `unified_vs_single_task_data.json` so a later
unified checkpoint can replace the current snapshot without changing plotting
code. The current unified paths point to the latest completed evaluations.

The current L-peg pair is not tolerance-matched: the legacy task-specific
teacher uses 0.5 mm while the selected unified teacher uses 1 mm.
