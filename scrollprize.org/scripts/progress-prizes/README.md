# Progress Prize form rollover

This automation prepares the next monthly Google Form, changes only the managed
deadline and responder link in `docs/34_prizes.md`, gates activation on tests and
the exact Vercel preview commit, then closes the old form before opening the new
one. The cutoff is midnight immediately after the last calendar day in
`America/Los_Angeles` (the published deadline remains 11:59pm Pacific).

The repository is public. Google authentication is therefore keyless: GitHub's
short-lived OIDC token is exchanged through Google Workload Identity Federation
(WIF). Do **not** add a service-account JSON key, OAuth refresh token, Google
cookie, form editor URL, Drive ID, folder ID, group address, or access token to a
repository secret, file, log, cache, or artifact.

## Workflow map

- `progress-prizes-google.yml` is the only Google-authenticated reusable job. It
  accepts only `workflow_dispatch` or `schedule`, checks the immutable repository
  and owner IDs, checks out trusted `main`, and exports only the intentionally
  public `responder-uri`.
- `progress-prizes-rehearsal.yml` is the July 20 staging rehearsal. Its fixed
  branches are `codex/progress-prize-smoke-20260720` and
  `codex/progress-prize-smoke-base-20260720`.
- `progress-prizes-page-pr.yml` updates the page, runs dependency-free tests,
  commits, and creates a draft PR. It has no Google configuration or OIDC.
- `progress-prizes-pr-safety.yml` runs for public PRs with read-only contents and
  no secret or OIDC access.
- `progress-prizes-vercel-preview.yml` runs trusted default-branch verifier code
  on a Vercel `repository_dispatch`. It never checks out or executes deployed
  branch code.
- Milestone 4 adds `progress-prizes-production.yml` only after the complete
  staging rehearsal passes. It provides guarded `validate`, `dry-run`, `prepare`,
  `activate`, and `verify` operations.
- Milestone 5 adds `progress-prizes-schedule.yml` only after production
  `validate` and `dry-run` pass. Reaching `main` is what enables its Pacific-time
  schedules.

The workflows invoke:

```text
node scrollprize.org/scripts/progress-prizes/automation-cli.mjs COMMAND ...
```

Private identifiers are read only from the protected Environment variables
below. Operation JSON is kept in `$RUNNER_TEMP`, never uploaded, and only a
canonical `forms.gle` or `docs.google.com/forms/d/e/.../viewform` URL may cross
the authenticated job boundary. Because GitHub variables are not masked
automatically, the workflows validate and register every protected Google and
Vercel identifier with `add-mask` before an action or verifier can log it.

## GitHub Environments

Create these environments before merging the workflows. Restrict all three to
the protected `main` branch. Require an authorized Vesuvius Challenge reviewer
on `progress-prizes-production`; this is the human gate before production Google
access, including the real July 31 activation. Staging and preview should not
require a month-end wait.

### `progress-prizes-staging`

Protected Environment variables:

- `GOOGLE_WORKLOAD_IDENTITY_PROVIDER`
- `GOOGLE_SERVICE_ACCOUNT_EMAIL`
- `PROGRESS_PRIZE_DRIVE_ID` (the staging Shared Drive)
- `PROGRESS_PRIZE_FOLDER_ID` (the staging active folder)
- `PROGRESS_PRIZE_ARCHIVE_FOLDER_ID` (the staging archive folder)
- `PROGRESS_PRIZE_SOURCE_FORM_ID` (the current live form's private file ID)
- `PROGRESS_PRIZE_EDITOR_GROUP_EMAIL` (a staging-only internal group)

The staging service account needs read/copy access to the current production
source, but must not be able to edit or share it. Give it the necessary creator,
editor, and sharing rights only in the staging Shared Drive. The staging group
must not be the production editor group.

### `progress-prizes-production`

Protected Environment variables:

- `GOOGLE_WORKLOAD_IDENTITY_PROVIDER`
- `GOOGLE_SERVICE_ACCOUNT_EMAIL`
- `PROGRESS_PRIZE_DRIVE_ID` (the production Shared Drive)
- `PROGRESS_PRIZE_FOLDER_ID` (the production active forms folder)
- `PROGRESS_PRIZE_SOURCE_FORM_ID` (the current live form's private file ID)
- `PROGRESS_PRIZE_EDITOR_GROUP_EMAIL` (the production internal editor group)

The production account must be able to copy, edit, and share the managed forms
in this folder. It should have no access to the staging Shared Drive. No archive
folder is supplied to production, and the workflow deliberately supplies empty
staging-identity values in production mode. `PROGRESS_PRIZE_SOURCE_FORM_ID` is
the initial fallback: after the first managed activation, Drive `appProperties`
discovery selects the prior cycle's managed target, so this private variable does
not need a monthly edit.

### `progress-prizes-preview`

Protected Environment variable:

- `VERCEL_PROJECT_ID`

Environment secret:

- `VERCEL_AUTOMATION_BYPASS_SECRET`

The bypass value is needed because the current Vercel previews are protected by
SSO. It is sent only after the verifier has accepted an HTTPS `*.vercel.app`
origin for the configured project; redirects are never followed. This is the
only long-lived GitHub secret required by these workflows, and it is a Vercel
preview secret—not a Google credential.

## Google WIF setup

Enable the Google Forms and Drive APIs. Create separate service accounts and
separate WIF providers (or equivalently isolated provider conditions) for staging
and production. Map these GitHub OIDC claims:

```text
google.subject                 = assertion.sub
attribute.repository          = assertion.repository
attribute.repository_id       = assertion.repository_id
attribute.repository_owner_id = assertion.repository_owner_id
attribute.ref                 = assertion.ref
attribute.event_name          = assertion.event_name
attribute.environment         = assertion.environment
attribute.job_workflow_ref    = assertion.job_workflow_ref
```

Use this condition blueprint for staging, substituting only the Environment name
for production:

```text
attribute.repository == 'ScrollPrize/villa' &&
attribute.repository_id == '890972577' &&
attribute.repository_owner_id == '121906140' &&
attribute.ref == 'refs/heads/main' &&
(attribute.event_name == 'workflow_dispatch' || attribute.event_name == 'schedule') &&
attribute.environment == 'progress-prizes-staging' &&
attribute.job_workflow_ref == 'ScrollPrize/villa/.github/workflows/progress-prizes-google.yml@refs/heads/main'
```

Bind only that provider principal to its matching service account with
`roles/iam.workloadIdentityUser`. Use the numeric Google project number when
constructing the `principalSet` member. Do not grant one provider impersonation
rights on both accounts. The workflow asks for a 900-second access token scoped
to read-only Forms/Drive scopes for `validate`, `verify`, and dry runs. Mutations use
`forms.body` plus `drive`: this headless workflow must find a pre-existing form
and app-property-managed files in a Shared Drive, which `drive.file` cannot
reliably authorize without an interactive picker. The separate service accounts
and Shared Drive ACLs bound the writable resources. Credential-file creation and
global environment export remain disabled.

Google file access is controlled separately by Shared Drive and form ACLs. WIF
impersonation alone must not grant the staging identity production write access.

## Repository and Vercel prerequisites

Before rehearsal, an administrator must:

1. Protect `main`, restrict direct pushes, require pull requests, and require the
   `Public no-secret tests` check. Keep squash merge enabled.
2. In **Settings → Actions → General**, allow the workflow token the requested
   write permissions and enable **Allow GitHub Actions to create and approve pull
   requests**. Without this, draft creation and ready/merge transitions fail
   closed. GitHub places `pull_request` checks triggered by a PR created with
   `GITHUB_TOKEN` into an approval-required state; approve the no-secret test run
   on each automation PR before the exact-commit gate expires.
3. Configure Vercel to send `repository_dispatch` events
   `vercel.deployment.ready` or `vercel.deployment.success`. The documented
   payload fields `environment`, `project.id`, `url`, `git.sha`, and `git.ref`
   are validated; the workflow run title is bound to `git.sha`.
4. Confirm the authenticated dispatch actor is exactly `vercel[bot]`. If the
   supported Vercel integration uses a different documented immutable GitHub App
   identity, update and review the allowlist and its contract test; never remove
   the actor check or accept a user token.
5. Confirm Vercel builds both fixed smoke branches and production rollover PR
   branches. The verifier associates the payload with GitHub: exactly one open
   automation PR with the exact head SHA/ref and allowed base, or the exact
   current SHA of the fixed smoke base after merge.

As observed on July 20, 2026, the repository still lacked the three Progress
Prize Environments, `main` branch protection, and the Actions PR setting; Vercel
previews redirected to SSO. Those are external blockers, not values that can be
safely filled in by repository code.

## July 20 rehearsal

The workflow and WIF policy intentionally require the foundation workflows to be
reviewed and present on protected `main` before Google authentication works.
After the environment setup above:

1. Dispatch **Progress Prize July 20 rehearsal** with `full-rehearsal` and the
   defaults. Its first Google operation is the required read-only production
   validation of the live July form; it cannot close, rename, publish, or change
   that form's permissions.
2. The preparation clock is July 25 (inside the seven-day window) and activation
   is August 1 at 07:01 UTC (just after the July Pacific cutoff).
3. The run copies July read-only into staging, injects one `after-copy` failure,
   resumes the same managed copy, prepares the closed August target, and opens a
   draft PR against the fixed ephemeral smoke base.
4. After the exact Vercel preview and public test pass, it injects one
   `after-close-source` failure, resumes activation, verifies source closed and
   target open, readies and squash-merges the smoke PR, verifies the smoke-base
   preview, and reruns activation to prove idempotency.
5. The full run unpublishes and closes both smoke forms, removes published-reader
   permissions, moves them into the staging archive, and verifies the cleaned
   state. Internal IDs never appear in its summary.
6. Keep both smoke branches for review. Delete them manually only after the audit
   is accepted. The production July form and `main` website content are not
   modified by this rehearsal.

Archived forms retain their managed environment/role/cycle markers. Discovery is
Shared-Drive-wide, so the fixed July smoke cycle cannot silently create a second
copy after cleanup. Repeating that exact cycle requires an explicit, reviewed
retirement of the archived smoke markers or a new rehearsal cycle/date.

Individual rehearsal segments are available for recovery. They remain fixed to
the same staging identities, folder, clock rules, and smoke branches; the core
also rejects fault injection unless every staging condition is true.

After the full rehearsal passes, add and merge the guarded production workflow,
then dispatch production `dry-run` and `validate`. Add
`progress-prizes-schedule.yml` only at the final schedule milestone: once it
reaches `main`, GitHub enables the Pacific daily schedules.

## Local verification

No package installation is required for the automation tests:

```bash
cd scrollprize.org
node --test "scripts/progress-prizes/**/*.test.mjs" "src/components/atlas/**/*.test.js"
git diff --check
```

Lint only the new workflows with actionlint (the repository has unrelated legacy
workflow findings):

```bash
actionlint ../.github/workflows/progress-prizes-*.yml
```

The rehearsal-foundation contract test checks immutable action pins, OIDC
isolation, repository IDs, staging-only controls, exact-commit Vercel
association, and trusted GitHub check provenance. The gated production and
schedule milestones have separate contract tests so those files cannot be
committed early merely to satisfy a foundation test.
