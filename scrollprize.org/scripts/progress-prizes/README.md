# Progress Prize form rollover

This automation prepares the next monthly Google Form, changes only the managed
deadline and responder link in `docs/34_prizes.md`, gates activation on tests and
the exact Vercel preview commit, then closes the old form before opening the new
one. The cutoff is midnight immediately after the last calendar day in
`America/Los_Angeles` (the published deadline remains 11:59pm Pacific).

A fresh copy is closed at the first possible Forms API call, before title, ACL,
or capability reconciliation. At cutoff, the fingerprint-bound target records
durable activation intent before the source is closed. Activation refetches and
revalidates both forms after the preview gate, so a manual source closure or
post-gate ACL/metadata change cannot be mistaken for a recoverable transition.

The repository is public. Google authentication is therefore keyless: GitHub's
short-lived OIDC token is exchanged through Google Workload Identity Federation
(WIF). Do **not** add a service-account JSON key, OAuth refresh token, Google
cookie, form editor URL, Drive ID, folder ID, group address, or access token to a
repository secret, file, log, cache, or artifact.

## Workflow map

- `.github/actions/progress-prizes-google/action.yml` is the shared local action
  used directly by every Google-authenticated job. It validates controls,
  exchanges OIDC without a credential file, and runs the dependency-free CLI.
- `progress-prizes-rehearsal.yml` is the July 20 staging rehearsal. Its Google
  jobs are ordinary top-level jobs with literal protected Environment bindings;
  its fixed branches are `codex/progress-prize-smoke-20260720` and
  `codex/progress-prize-smoke-base-20260720`.
- `progress-prizes-page-pr.yml` updates the page, runs dependency-free tests,
  commits, and creates a draft PR. It has no Google configuration or OIDC.
- `progress-prizes-pr-safety.yml` runs for public PRs with read-only contents and
  no secret or OIDC access.
- `progress-prizes-vercel-preview.yml` runs trusted default-branch verifier code
  on a Vercel `repository_dispatch`. It never checks out or executes deployed
  branch code.
- `progress-prizes-production.yml` provides guarded `validate`, `dry-run`,
  `prepare`, `activate`, and `verify` operations after the complete staging
  rehearsal has passed. Every authenticated job is literal in that top-level
  workflow and calls the same local action exercised by staging.
- `progress-prizes-schedule.yml` is the secret-free scheduler added only after
  production `validate` and `dry-run` passed. Reaching `main` enables its
  Pacific-time schedules; a manual dispatch is permanently restricted to a
  read-only production dry-run.

The workflows invoke:

```text
node scrollprize.org/scripts/progress-prizes/automation-cli.mjs COMMAND ...
```

Private identifiers are read only from protected Environment secrets below.
They are configuration values rather than Google credentials, but secrets are
required because GitHub does not automatically redact Environment variables
from public Actions logs. Operation JSON is kept in `$RUNNER_TEMP`, never
uploaded, and only a canonical `forms.gle` or
`docs.google.com/forms/d/e/.../viewform` URL may cross the authenticated job
boundary. The workflows also register every protected identifier with
`add-mask` before validation as defense in depth.
Google-secret-consuming jobs are never placed behind `workflow_call`, and no
workflow uses `secrets: inherit`. Live July validation runs showed that GitHub
created and approved the correct deployment but still evaluated every protected
Environment secret as empty inside a called workflow, both with expression-based
and literal Environment names. Each top-level workflow that authenticates to
Google therefore binds its Google jobs directly to a literal protected
Environment and invokes the same local action. The separate scheduler
authenticates only to GitHub and dispatches the production workflow. The exact
trigger commit—not a mutable branch tip—is the executable code that receives
the approved secrets.

## GitHub Environments

Create these environments before merging the workflows. Restrict all four to
the protected `main` branch. Require an authorized Vesuvius Challenge reviewer
only on `progress-prizes-production-activation`; this is the human gate for the
real close/open transition. The approval job receives no Google configuration,
OIDC permission, or repository write permission. Staging, preview, and
`progress-prizes-production` itself must not require a reviewer, so a queued
daily preparation cannot hold the production concurrency lock and block the
cutoff or its recovery run.

### `progress-prizes-staging`

Protected Environment secrets:

- `GOOGLE_WORKLOAD_IDENTITY_PROVIDER`
- `GOOGLE_SERVICE_ACCOUNT_EMAIL`
- `PROGRESS_PRIZE_STAGING_SERVICE_ACCOUNT_EMAIL` (an independently configured
  copy of the expected staging service-account identity)
- `PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL` (one private human kept as the inherited
  break-glass Shared Drive Manager)
- `PROGRESS_PRIZE_DRIVE_ID` (the staging Shared Drive)
- `PROGRESS_PRIZE_FOLDER_ID` (the staging active folder)
- `PROGRESS_PRIZE_STAGING_FOLDER_ID` (an independently configured copy of the
  expected staging active-folder ID)
- `PROGRESS_PRIZE_ARCHIVE_FOLDER_ID` (the staging archive folder)
- `PROGRESS_PRIZE_SOURCE_FORM_ID` (the initial owner-My-Drive form's private file ID)
- `PROGRESS_PRIZE_EDITOR_GROUP_EMAIL` (the staging-only group containing the
  three internal form editors)

Share only the initial form, not its My Drive folder, directly with the staging
service account using a Drive `reader` permission. The current Forms sharing UI
offers only **Responder** and **Editor**, so it cannot create this least-privilege
ACL: use Drive API `permissions.create` with the following generic request and
keep the resulting permission non-expiring:

```text
fileId: <INITIAL_FORM_FILE_ID>
supportsAllDrives: true
sendNotificationEmail: false
requestBody: {type: "user", role: "reader", emailAddress: <STAGING_SERVICE_ACCOUNT_EMAIL>}
```

Do not substitute the Forms **Responder** role. The preflight requires `canCopy`
and rejects `canEdit` or `canShare`; it also verifies the direct reader ACL.
Give the account creator, editor, and sharing rights only in the staging Shared
Drive. The staging group and production group are distinct private groups, each
containing the same three internal editors. Staging bootstrap deliberately does
not copy any production collaborator: the staging form receives only the
staging group and its anonymous published-responder permission.
Keep exactly two editable members on the staging Shared Drive: the staging
automation account and `PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL`, both at the
non-expiring Manager level. Their permissions must therefore appear as inherited
`organizer` permissions on the active folder and managed forms. The break-glass
identity must be a user, not a group, and it must not have a second direct
folder/form permission. Do not add the staging editor group to the Shared Drive
or active folder; the automation grants that group an explicit writer permission
on each managed staging form.

### `progress-prizes-production`

Protected Environment secrets:

- `GOOGLE_WORKLOAD_IDENTITY_PROVIDER`
- `GOOGLE_SERVICE_ACCOUNT_EMAIL`
- `PROGRESS_PRIZE_STAGING_SERVICE_ACCOUNT_EMAIL` (the staging reader identity
  expected on the initial live form during read-only validation)
- `PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL` (one private human kept as the inherited
  break-glass Shared Drive Manager)
- `PROGRESS_PRIZE_DRIVE_ID` (the destination/managed production Shared Drive)
- `PROGRESS_PRIZE_FOLDER_ID` (the destination active forms folder)
- `PROGRESS_PRIZE_SOURCE_FORM_ID` (the initial owner-My-Drive form's private file ID)
- `PROGRESS_PRIZE_EDITOR_GROUP_EMAIL` (the production-only group containing the
  three internal form editors)

Share only the initial form, not its My Drive folder, directly with the
production service account as Editor. The preflight requires a direct,
non-expiring writer ACL plus `canCopy` and `canEdit`; `canShare` is deliberately
not required on this owner-controlled source. Share the form with the production
editor group as Editor, remove the three internal editors' individual form ACLs
after group access is verified, and retain the external editor as one direct,
non-expiring production writer. Do not add that external editor to either
internal group or to staging. Production copies preserve the configured
production group and that direct external writer across cycles; neither service
account is recreated as a direct form collaborator.

The production account must separately be able to create, copy, edit, and share
forms in the production destination folder, and must have no access to the
staging Shared Drive. The destination is validated as a writable folder in the
configured Shared Drive before any copy. Keep exactly the production automation
account and `PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL` as non-expiring Shared Drive
Managers, so both appear as inherited `organizer` permissions on the active
folder and managed forms. The break-glass identity must be a user, not a group,
and it must not also be a direct form/folder collaborator. Do not add the
production editor group to the Shared Drive or active folder; it is granted
explicitly on every managed form. Every active permission role is inspected:
any other Google service account—including a reader or owner—fails closed, as
do inherited editors, domains, additional Managers, or Content managers.

No archive or staging folder is supplied to production. Production mutation
jobs receive no staging identity. The initial July read-only validation is the
only exception: the workflow conditionally supplies the expected staging reader
identity for that source cycle, so it can verify the live form without granting
the production identity access to staging. Later validation cycles receive an
empty staging-identity input.
`PROGRESS_PRIZE_SOURCE_FORM_ID` is a one-time, explicit fallback. The July form
never receives managed environment/role/cycle markers and is never moved or
renamed. At activation it is closed first and receives only the private recovery
state marker. The August copy is created in the production Shared Drive and is
cryptographically bound to that exact source ID. After its activation, managed
`appProperties` discovery always selects the prior managed target, so the
fallback secret does not need a monthly edit and is ignored for later cycles.

### `progress-prizes-production-activation`

This Environment contains no secrets or variables. Require one authorized
Vesuvius Challenge reviewer, disallow self-review if the organization supports
it, and restrict deployment branches to protected `main`. The workflow first
freezes and verifies the exact Progress Prize PR, public test, and Vercel
preview; then this Environment records the human approval. The following
Google job rechecks the immutable lease immediately before mutation, so a PR or
`main` change while approval is pending fails closed.

### `progress-prizes-preview`

Protected Environment secret:

- `VERCEL_PROJECT_ID`

- `VERCEL_AUTOMATION_BYPASS_SECRET`

The bypass value is needed because the current Vercel previews are protected by
SSO. It is sent only after the verifier has accepted an HTTPS `*.vercel.app`
origin for the configured project; redirects are never followed. This bypass
value is the only secret here that grants access outside GitHub. The Google
Environment secrets contain identifiers and ACL configuration only—not a key,
refresh token, or other reusable credential.

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
attribute.workflow_ref        = assertion.workflow_ref
attribute.workflow_sha        = assertion.workflow_sha
```

Use this condition blueprint for staging. Staging has no scheduled caller, so it
accepts manual dispatch only:

```text
attribute.repository == 'ScrollPrize/villa' &&
attribute.repository_id == '890972577' &&
attribute.repository_owner_id == '121906140' &&
attribute.ref == 'refs/heads/main' &&
attribute.event_name == 'workflow_dispatch' &&
attribute.environment == 'progress-prizes-staging' &&
attribute.workflow_ref == 'ScrollPrize/villa/.github/workflows/progress-prizes-rehearsal.yml@refs/heads/main' &&
assertion.workflow_sha == assertion.sha
```

Use this separate condition blueprint for the production provider after the
production workflow reaches protected `main`:

```text
attribute.repository == 'ScrollPrize/villa' &&
attribute.repository_id == '890972577' &&
attribute.repository_owner_id == '121906140' &&
attribute.ref == 'refs/heads/main' &&
attribute.event_name == 'workflow_dispatch' &&
attribute.environment == 'progress-prizes-production' &&
attribute.workflow_ref == 'ScrollPrize/villa/.github/workflows/progress-prizes-production.yml@refs/heads/main' &&
assertion.workflow_sha == assertion.sha
```

The schedule milestone does not receive Google configuration or OIDC. It
computes the Pacific window and dispatches this exact production workflow with
`GITHUB_TOKEN`; GitHub documents `workflow_dispatch` as an event that is allowed
to create a new run from `GITHUB_TOKEN`. The production WIF condition therefore
does not need to permit the schedule workflow path or a `schedule` event.

Bind only that provider principal to its matching service account with
`roles/iam.workloadIdentityUser`. Use the numeric Google project number when
constructing the `principalSet` member. Do not grant one provider impersonation
rights on both accounts. The workflow asks for a 1200-second access token scoped
to read-only Forms/Drive scopes for `validate`, `verify`, and dry runs. Mutations
use `forms.body` plus `drive`: this headless workflow must find a pre-existing
form and app-property-managed files in a Shared Drive, which `drive.file` cannot
reliably authorize without an interactive picker. The separate service accounts,
the two exact-file ACLs on the initial My Drive form, and the isolated Shared
Drive ACLs bound the writable resources. Credential-file creation and global
environment export remain disabled.

Google file access is controlled separately by Shared Drive and form ACLs. WIF
impersonation alone grants no Drive access. The staging identity has read/copy
access only to the initial production form; the production identity has
copy/edit access only to that form plus its managed destination. Never share the
owner's My Drive folder, enable domain-wide delegation, or substitute a JSON key
or user refresh token if Workspace policy blocks service-account access.

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

The July 20 setup created the staging, production, and preview Environments,
enabled an active `Protect main` ruleset, and verified an exact Vercel preview
through its protected bypass secret. The separate secret-free production
activation Environment is added with the guarded production milestone. These
controls remain administrator-managed external configuration; their private
values never belong in repository code.

## Initial My Drive cycle

The current July form may remain owner-controlled in My Drive. This is a narrow
bootstrap exception, not a second managed storage location. The code permits an
explicit fallback only for the immutable `2026-07` source cycle; a missing
managed source in any later cycle fails instead of reusing July:

1. Put the same private form ID in the staging and production protected
   Environments. Do not put it in a repository file or ordinary GitHub variable.
2. Give the staging service account the direct Drive API `reader` permission
   described above and give the production service account a direct Forms
   Editor/Drive `writer` permission. Keep both ACLs non-expiring through the
   rehearsal; keep the production writer through activation, recovery, and
   active verification. The staging reader is permitted only on this explicit
   initial My Drive source, never on a managed form or destination folder.
3. Create separate private production and staging editor groups containing the
   same three internal editors. Share July with the production group as Editor,
   remove those three people's direct ACLs after verification, and keep the one
   external editor as a direct production writer. Do not share either the
   production group or the external editor with staging.
4. The Google copy operation does not carry the source ACL into the destination.
   The destination first inherits its Shared Drive access, then the automation
   reconciles the anonymous published-reader permission and the environment's
   intended collaborators. Staging copies only its staging group. Production
   copies its production group and direct external writer. Owner and
   automation-service-account ACLs are never recreated on a copy.
   Effective inherited writers/commenters and Shared Drive administrative roles
   are checked exactly. Managed resources must expose exactly one inherited
   Manager permission for the current automation account and one for the
   configured break-glass user. Each permission must have exactly one Drive role
   source: a Shared Drive `member` organizer inherited from the configured Drive
   itself. A merged direct file/folder grant or any other role source fails. The
   break-glass permission is ignored in form-ACL equality only after its identity,
   role source, inheritance, uniqueness, and lack of expiration are verified.
   Any other service account at any role, inherited editor, domain writer,
   Manager, or Content manager fails closed.
5. Before the real cutoff, repeat the close/open mutation path on a sacrificial
   My Drive form. Read-only capability validation cannot prove that Workspace
   policy will allow the service account to call Forms publish settings and
   write the private recovery marker.
6. After the August form is active and verified, the owner may remove both
   direct service-account ACLs from July. Later cycles resolve only managed forms
   in the production Shared Drive, even if the stale fallback secret remains.

If Workspace policy refuses direct sharing to a service-account principal, move
the source into the production Shared Drive or redesign the identity boundary.
Do not work around that policy with domain-wide delegation or reusable Google
credentials.

## July 20 rehearsal

The workflow and WIF policy intentionally require the foundation workflows to be
reviewed and present on protected `main` before Google authentication works.
After the environment setup above:

1. Dispatch **Progress Prize July 20 rehearsal** with `full-rehearsal` and the
   defaults. Its first Google operation is the required read-only production
   validation of the live July My Drive form; it cannot close, rename, publish,
   move, or change that form's permissions.
2. The preparation clock is July 25 (inside the seven-day window) and activation
   is August 1 at 07:01 UTC (just after the July Pacific cutoff).
3. The run uses the staging identity to copy July read-only from My Drive into
   the staging Shared Drive, injects one `after-copy` failure, resumes the same
   managed copy, prepares the closed August target, and opens a draft PR against
   the fixed ephemeral smoke base.
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

The full rehearsal passed through verified cleanup before the production
workflow was added. The guarded workflow was then merged, the production WIF
condition was restricted to its exact path, and production `validate` and
`dry-run` both passed on July 21, 2026. The scheduler is the final milestone:
GitHub enables its Pacific schedules only after that reviewed file reaches
`main`.

## Production operations and recovery

Use **Progress Prize production rollover** on `main`. For July to August, keep
`source-cycle=2026-07` and `target-cycle=2026-08`; later targets must always be
the immediately following month. Leave `request-id` empty for every manual run.

- `validate` performs the read-only live-form, capability, publishing, response,
  ACL, copy, and linked-Sheet preflight. It never writes Google or GitHub state.
- `dry-run` is deliberately useful before the normal seven-day preparation
  window. It uses the real clock and the same preparation preflight, but extends
  only the read-only planning window to 31 days. It creates no copy, changes no
  ACL or publishing state, writes no website branch, and records only the
  proposed public title and deadline in the run summary. Real `prepare` remains
  fixed to seven days.
- `prepare` is safe to dispatch repeatedly. It succeeds without opening a page
  PR outside the seven-day window. Inside the window it resumes the one managed
  target for the cycle, keeps it published but closed, and reconstructs the one
  marker-only draft page PR.
- `verify` with `prepared` requires that exact page-only PR on current `main`;
  `active` requires the completed website and Google close/open state.
- `activate` should be dispatched near 23:40 Pacific on the final day. The exact
  PR tests and Vercel preview pass first. Approval is offered only when the real
  cutoff is at most one hour away (or has passed), through the secret-free
  `progress-prizes-production-activation` Environment. The job waits without a
  Google token, authenticates at cutoff, reacquires a zero-wait GitHub lease,
  closes the source, opens and reload-verifies the target, then merges only the
  activated commit.

If preparation, activation, or merge stops, rerun the same operation and cycle.
Managed Drive markers make Google copy and close/open recovery idempotent. A
rerun after a completed merge performs read-only active verification. If `main`
moved before mutation, activation reconstructs and rechecks a stale-parent page
commit; any multi-file, merge, wrong-path, or query-bearing change fails closed.
Never use simulated time, fault controls, alternate branches, or staging folders
for production; those inputs are absent and the shared action rejects them
before authentication.

## Automated schedule and immediate smoke

The scheduler owns no Google or Vercel configuration, protected Environment,
OIDC permission, reusable credential, or repository secret. It runs only trusted
code from the exact `main` commit, derives cycles from the real
`America/Los_Angeles` clock, and uses the repository `GITHUB_TOKEN` solely to
dispatch `progress-prizes-production.yml` on `main`. The dispatch response is
bound to the exact child run ID and public Actions URL.

- `06:17` Pacific is the daily preparation probe. It no-ops before the exact
  seven-day window and after cutoff. Once an exact page-only draft PR already
  sits directly above current `main`, it skips repeated preparation instead of
  force-pushing a new commit and restarting checks.
- `23:40` Pacific on candidate final days dispatches activation only when the
  observed date is the actual last calendar day. The production workflow—not
  the scheduler—runs tests and the Vercel gate, requests human approval, waits
  for cutoff without a Google token, then authenticates.
- `00:17` and `06:47` Pacific on the first day are independent recovery probes.
  Delayed final-day events retain the previous source cycle through that first
  day, while every day-two event no-ops. If an exact production run is already
  nonterminal, the scheduler does not enqueue a stale duplicate.

Production and scheduler concurrency groups never cancel an in-progress run and
use GitHub's queued concurrency mode so a manual race cannot silently replace a
pending cutoff run. The scheduler itself never sleeps. GitHub may delay or drop
a scheduled event, and public-repository schedules can be disabled after 60 days
without repository activity, so the production workflow keeps its manual
`prepare` and `activate` recovery path.

Immediately after the scheduler reaches `main`, manually dispatch **Progress
Prize production schedule** once. Manual scheduler runs have no inputs and can
only dispatch the real-clock production `dry-run`; they cannot prepare, close,
open, share, publish, merge, or update the website. Verify that the scheduler's
recorded child run is the exact successful read-only production run. Scheduled
preparation must remain reviewer-free; the reviewer gate belongs only to the
secret-free `progress-prizes-production-activation` Environment.

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
actionlint \
  -ignore 'unexpected key "queue" for "concurrency" section' \
  ../.github/workflows/progress-prizes-*.yml
```

The narrow ignore is for actionlint 1.7.12, released before GitHub added the
valid `concurrency.queue: max` syntax in May 2026. It suppresses only that known
schema-lag diagnostic; every other workflow finding remains fatal. Remove the
ignore once a released actionlint understands queued concurrency.

The rehearsal-foundation contract test checks immutable action pins, OIDC
isolation, repository IDs, staging-only controls, exact-commit Vercel
association, and trusted GitHub check provenance. The gated production and
scheduler have separate executable contract tests for the clock boundaries,
minimal permissions, deduplication, fixed dispatch endpoint, child-run binding,
redacted failures, and absence of Google configuration.
