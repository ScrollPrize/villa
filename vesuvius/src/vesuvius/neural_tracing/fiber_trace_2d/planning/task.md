# Trace2CP Metric Stdout And Presence Visualization

Small Trace2CP runner fixes:

- Print the selected single-pair `trace2cp_error` as its own stdout line, with
  diagnostics moved to the following details line.
- When `--trace2cp-use-presence` is active, include the sheet/fiber presence
  probability map in Trace2CP visualizations.
