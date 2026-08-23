# Task: resolve published Lasagna normals for managed Fiberlet jobs

Make `las_manager fiberlet run <fiber-inference>` sufficient for the normal
managed workflow. The manager must discover a compatible regular Lasagna
normal prediction for the Fiber inference's sample, volume, and compatible
base coordinate frame from the cached public open-data catalogue, refresh the catalogue under
the manager's existing age policy, and pass the published normal manifest to
the native Fiberlet processor through its read-through cache.

Keep the existing explicit local-normal-run form available for testing and
overrides. Never substitute Fiber direction `nx`/`ny` for regular Lasagna
`grad_mag`/`nx`/`ny`
surface normals. If no compatible normal prediction exists, or selection is
ambiguous, fail before creating a managed run and report the candidates or the
missing requirement clearly. Completion must expose both local Fiber runs and
published normal representations.
