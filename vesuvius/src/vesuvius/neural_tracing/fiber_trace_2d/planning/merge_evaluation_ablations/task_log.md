# Merge log

Source be93b129a, destination83dad0268/fiber-vischeck. Seven conflicted files: task/task_plan/status/task_log kept ours exactly; changelog/specs combined additive sections; live_evaluate retains producer provenance + progress reporting together with demand GPU lease/heartbeats.
Auto-merge semantic fix: incoming implementation-hash whitelist in cached extension/sweeps conflicts with existing parameter-only cache identity and crashes on new hash-free model metadata. Replaced both gates with shared inference_cache.validate_cached_model (checkpoint,explicit inference parameters), tests cover legacy implementation field and changed numerical params. Removed obsolete whitelist module/test, updated docs; no cache metadata changes. Retained shared mmap publication and incoming deferredGPU lifecycle.
Native embedding normalized-mean/coherence, heartbeat and training-span eligibility retained. Updated stale specs restriction on nondefault ellipse rendering to match current supported implementation.

Review follow-ups: adaptive heartbeat gated on native still running during background draining; cache extension resolves omitted parameters from recorded model and treats precision=auto as recorded effective precision. Prevents silent default batch-size/embedding-output drift; numerical mismatches still fail. Added default/auto regression assertions.
Expanded approved workspace build/test runners to include native tracer embedding and incoming demand GPU/training-span/embedding Python regressions. No training or evaluation started during merge.
Follow-up user request: after merge completion freeze finalEMA999 from stopped latest training and run full50fiber eval+vis. Do not select snapshot until merge is complete.

Validation complete: approved workspace native build succeeded. Approved replay runner: 23 replay/render cases, 79 native tracer cases, 206 Python tests passed, 8 skipped (71.60s). Independent review follow-ups resolved as above; live GPU lease reacquisition is not exercised by this CPU suite. Protected four planning files are byte-identical to destination HEAD; git diff --check clean. Changelog histories fused rather than choosing one side.
