# Training trace execution contract

- Reviewed trace evaluation uses the same prepared workload, precision,
  preprocessing, products, native tracing, and metrics as standalone evaluation.
- Managed training lends its resident raw-model GPU replicas at a completed
  test boundary. Optimizer and EMA tensors remain resident and unchanged.
- Evaluation does not spawn a replacement Python benchmark or model process.
  Native tracing and existing parallel CPU loading/accumulation remain shared.
- Manager polling continues during evaluation. Revocation stops new inference,
  joins borrowed GPU work and cleans shared buffers before ordinary GPU drain.
  The exact pending test survives reallocation and retries before more training.
- Dense/raw/EMA metrics keep their existing sample-step conventions. The
  independent raw best-by-restarts checkpoint retains exact frozen provenance.
- Debug unmanaged training must explicitly disable reviewed tracing; no hidden
  extra-model fallback is permitted. A resident test cannot substitute a second
  checkpoint for its embedding channels.
