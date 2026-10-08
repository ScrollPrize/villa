# Final EMA999 evaluation log

Merge completed afb14b400 (parents 83dad0268, be93b129a). Independent final-eval plan/config review approved unchanged protocol; immutable snapshot pinned before launch.

Final stopped-training EMA999: update 11800, samples 1510400, test_ema_0p999/loss_total=0.29991902176998864. SHA256 4d7c3b7ed06a29a344f8f21175158bf40c42ade440943ff68aa2775ee0cc1d82. Snapshot provenance in output_root/snapshot.json.

Config: fiber_trace_3d/configs/eval_eligible_ema999_final_vis_20260914_v1.json.
Output: /ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1.
Approved runner /home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py with freeze-snapshot, prepare, validate-full, full and --config pointing to above absolute config. Preparation and validation passed. Full run launched with render=true, all8GPUs exclusive priority100, embedding off.

Completed exit0: all180traces and180renders, 50fiber index README.md/per_fiber.json/per_fiber.csv written. Native wall1413.538s including rendering (not a trace-only speed benchmark). Same6867.627253mm coverage.113reported restarts, mean60.242344mm, median9.337232, P05=.232563, P95=94.960035. Per-scroll restart counts0139=50,0332=4,1299=7,1451=0,Paris4=52. Validated complete_report against final summary, every rendered page exists/nonempty, index counts match, no incomplete markers. Visually inspected case45 combined CT/presence, distinct top/side, labels and sections.

Compared with u10200:119→113restarts (−5.04%), mean57.230→60.242mm (+5.26%). Not uniform:0332/1299 regress; median/P05 and equal-scroll score lower (69.041084→61.052591). Classic292/23.439mm. Full five-model global/per-scroll statistics saved in docs/benchmarks/eligible_ema999_final_u11800_20260914.{md,json}. All8GPUs exclusive manager task9be2195d-3854-41e6-b6af-a0e67bc8465e; inference+vis completed normally. No algorithm changes or skipped requested evaluation cases.
