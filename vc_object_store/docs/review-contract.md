# vcobj review contract

Review by invariant, not just by file. A file marked inspected does not establish that its callers obey any particular contract.

1. Freeze the review commit. For each invariant, name the owning API and enumerate every entry point that produces, translates, stores, or consumes the relevant state. Follow native, Python, hosted, GUI, CLI and workflow callers as applicable. Record paths and symbols, including indirect callers and justified exclusions.
2. For each caller, record whether it delegates, performs legitimate boundary adaptation, duplicates policy, or remains unchecked. Check read and write counterparts together. Python bindings adapt language types; native code owns schema interpretation, validation and serialization. Likewise, store placement is resolved by the shared store layer.
3. Run the same edge case through sibling entry points. Apply local/referenced, sharded/packed, history on/off and current/exact cases where relevant. Record which cases are verified at runtime, which are source-traced, and which remain untested. Do not close a contract because one caller or an unrelated suite passes.
4. Trace each finding against the pre-fix commit before labeling it a regression. Review the fix diff separately, and reopen every affected invariant and caller. Report old omissions separately from regressions; counts of files or findings are not estimates of the fraction of defects found.

See the [store API contract](coherent-store-api-cleanup.md#contract) and [R5 inventory](review-r5/README.md). Fix and regression status belongs alongside the corresponding finding IDs.
