# Current snapshot test-loss versus tracing comparison

Freeze current EMA0.999, current EMA0.99, current raw from active combined training,
then evaluate in that order on existing50fiber/100direction workload. Record exact
samples/step/testloss/hash. Training staysalive with manager pauses only.
After EACH evaluation print full per-scroll tables against Classic and latest
best99/best999 pair plus completed current entries. Highlight best/ties; each
scroll20% overallweight. Omit pooledscore. Compare testloss to tracing performance
without claiming a reliable correlation from this small set. Preserve artifacts.
