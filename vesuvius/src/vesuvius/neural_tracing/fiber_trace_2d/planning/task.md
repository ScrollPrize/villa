# Neural follower FW/BW comparison

Commit prior results, then test the established FW/BW consistency filter on Flow v5 and Unified neural followers. Preserve checkpoint confidence/commit/gate policy and parallel GPU0–5 execution. Compare unfiltered controls, geometric errors, confidence stops, consistency stops, retained coverage and piece lengths.

Follow-up: also run neural memory and Flow v4 with the same filter. Include our own tuned EMA999 u12100 without FW/BW. Preserve old neural model policies and throughput settings; verify legacy checkpoint compatibility. Where upstream changes prevent a clean historical comparison, include fresh unfiltered controls.
