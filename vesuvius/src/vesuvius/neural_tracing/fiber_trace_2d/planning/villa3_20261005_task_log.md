# Tiled seam repair

Existing 2048 run completed: 8 local tiles, 7322 deduplicated lines, 3722 core fragments; 739 candidate links (715 unique, 24 ambiguous). Internal endpoints 2439, unmatched 985. Full run 4775.68s including 2347.49s anchor preparation; assembly 9.40s. Results remain in /ephemeral/hendrik/fiber_crop_bench/20261004_iso20_tiled2048_v2/results.json.

Current task reuses those saved deduplicated artifacts only. No inference, tracing or dedup rerun. Existing large dirty worktree preserved. Review requested for ambiguity semantics before changing BP behavior.

User requested report first, then a commit of current code/docs for integration.
No seam/BP code changes were made. Independent review: remove both percentage
gates for absolute support; ambiguous links require grouped alternatives, not
independent hard/soft equality edges. This remains pending, not a completed fix.
Completed tiled-run statistics and limitation are archived in tiled-tracing.md
and changelog.md. Commit-only checks: source/docs diff whitespace check passes
excluding generated SVG plots, which retain generator-produced trailing spaces.
No build or evaluation rerun is required by this documentation/checkpoint action.
