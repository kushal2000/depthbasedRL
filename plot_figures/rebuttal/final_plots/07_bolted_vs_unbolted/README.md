# Bolted versus unbolted solo policies

`bolted_vs_unbolted.py` compares early-drop-filtered insertion success for
policies trained and evaluated in matching fixture regimes. This is deliberately
different from the existing zero-shot
fixture-transfer plot, which evaluates a bolted-trained policy after merely
freeing its fixture.

The bolted baseline paths and future unbolted result globs are configured in
`bolted_vs_unbolted_data.json`. Missing unbolted offline evaluations render as
outlined `pending` bars; rerunning the script automatically selects the newest
matching JSON once those evaluations exist.
