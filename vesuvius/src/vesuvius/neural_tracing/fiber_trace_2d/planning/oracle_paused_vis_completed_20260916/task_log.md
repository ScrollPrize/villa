# GT-constrained fiberlet cost diagnostic

User requests cheapest valid routes and cost comparison for five weighted mini5 failures. Existing graph replay uses exact cost-bounded finite lookahead but unconstrained candidates. Diagnostic will reuse successor/evaluator code, retain path-dependent state and require real reference progress. Defaults unchanged. Prior mini5 task archived.

User redirected to visualization first before inspecting feasibility of these
cases. Oracle paused before implementation. Short benchmark binding forced render
off; added opt-in boolean forwarding and same CT base-volume binding as live eval,
reusing existing native renderer. Added regression assertions. Dedicated vis config
reuses five/model/weighted construction and inference cache, new output root.

Independent review of vis plan passed. Native renderer unchanged, no build needed.
Approved regression suite running; approved benchmark-full vis run launched.
Saved Classic/best beam comparison:pieces000,001,019 have1restart in both;
025,026 have0Classic/1best. All5weighted fiberlet1restart; all100%coverage.
Recorded table in mini5 benchmark doc. Oracle remains paused for inspection.

Vis completed exit0 in25.688608s. Verified five readable JPEGs plus render.json
with restart/cross-section metadata. All five replay and reference geometry JSON
values exactly equal non-vis pilot. Inspected Paris4 piece026 image: expected
CT/presence/reference/traced/top/side/restart/cross-section rows present.
Approved tests passed125native+263Python,8skips(86.58s). No native changes.
Oracle remains explicitly deferred pending user inspection; no optimal-route
results claimed. Review noted inherited anchor-backend render guard edge case;
current config explicitly fiberlet and native already rejects anchor rendering.
