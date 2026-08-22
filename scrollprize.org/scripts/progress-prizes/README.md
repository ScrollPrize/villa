# Progress Prize form rollover

This automation prepares the next monthly Google Form, changes only the managed
deadline and responder link in `docs/34_prizes.md`, gates activation on tests and
the exact Vercel preview commit, then closes the old form before opening the new
one. The cutoff is midnight immediately after the last calendar day in
`America/Los_Angeles` (the published deadline remains 11:59pm Pacific).

Each managed monthly form also has one distinct private Google Spreadsheet.
The Form is the canonical response store; the automation reads responses with
the Forms Responses API and appends previously unseen response IDs to that
month's spreadsheet. It never updates, clears, sorts, deletes, or replaces a
response row. Before opening the next form, activation closes and performs a
final append-only sync of the old form, then opens the already prepared target.
An existing Google Forms-linked Sheet is detected and left completely
untouched; the automation never reads its cells, changes its link, moves it, or
deletes it.

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
- `progress-prizes-page-pr.yml` updates the page, runs dependency-free tests,
  commits, and creates a draft PR. It has no Google configuration or OIDC.
- `progress-prizes-pr-safety.yml` runs for public PRs with read-only contents and
  no secret or OIDC access.
- `progress-prizes-vercel-preview.yml` runs trusted default-branch verifier code
  on a Vercel `repository_dispatch`. It never checks out or executes deployed
  branch code.
- `progress-prizes-production.yml` provides guarded `validate`, `dry-run`,
  `prepare`, `sync-responses`, `activate`, `reconcile-active`, and `verify`
  operations. Every authenticated job is literal in that top-level workflow.
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

Create these environments before merging the workflows. Restrict all three to
the protected `main` branch. Require an authorized Vesuvius Challenge reviewer
only on `progress-prizes-production-activation`; this is the human gate for the
real close/open transition. The approval job receives no Google configuration,
OIDC permission, or repository write permission. Preview and
`progress-prizes-production` itself must not require a reviewer, so a queued
daily preparation cannot hold the production concurrency lock and block the
cutoff or its recovery run.

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

Enable the Google Forms, Drive, and Sheets APIs. Create the production service
account and a WIF provider with the following mapped GitHub OIDC claims:

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

Use this condition blueprint for the production provider after the production
workflow reaches protected `main`:

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
to create a new run from `GITHUB_TOKEN`. Production Google authentication
therefore remains confined to the production workflow. The production WIF
condition does not need to permit the schedule workflow path or a `schedule`
event.

Bind only that provider principal to its matching service account with
`roles/iam.workloadIdentityUser`. Use the numeric Google project number when
constructing the `principalSet` member. Do not grant one provider impersonation
rights on both accounts. The workflow asks for a 1200-second access token.
`validate`, `verify`, and dry runs use read-only Forms body/response, Drive, and
Sheets scopes. Mutations use `forms.body`, read-only `forms.responses`, `drive`,
and `spreadsheets`: this headless workflow must find pre-existing forms and
app-property-managed files in a Shared Drive, which `drive.file` cannot reliably
authorize without an interactive picker. The Forms response scope is always
read-only. The separate service accounts, the two exact-file ACLs on the initial
My Drive form, and the isolated Shared Drive ACLs bound the writable resources.
Credential-file creation and global environment export remain disabled.

Google file access is controlled separately by Shared Drive and form ACLs. WIF
impersonation alone grants no Drive access. The production identity has copy/edit
access only to the initial form plus its managed destination. Never share the
owner's My Drive folder, enable domain-wide delegation, or substitute a JSON key
or user refresh token if Workspace policy blocks service-account access.

## Repository and Vercel prerequisites

Before enabling the production automation, an administrator must:

1. Protect `main`, restrict direct pushes, require pull requests, and require the
   `Public no-secret tests` check. Keep squash merge enabled.
2. In **Settings → Actions → General**, allow the workflow token the requested
   write permissions and enable **Allow GitHub Actions to create and approve pull
   requests**. Without this, draft creation and ready/merge transitions fail
   closed. GitHub places `pull_request` checks triggered by a PR created with
   `GITHUB_TOKEN` into an approval-required state; approve the no-secret test run
   on each automation PR before the exact-commit gate expires.
3. Configure Vercel to send the `repository_dispatch` event
   `vercel.deployment.ready`. The workflow runs only for a
   `codex/progress-prize-YYYY-MM` automation branch; all other Vercel
   deployments are ignored before a runner or protected Environment is
   requested. The documented payload fields `environment`, `project.id`, `url`,
   `git.sha`, and `git.ref` are validated; the workflow run title is bound to
   `git.sha`.
4. Confirm the authenticated dispatch actor is exactly `vercel[bot]`. If the
   supported Vercel integration uses a different documented immutable GitHub App
   identity, update and review the allowlist and its contract test; never remove
   the actor check or accept a user token.
5. Confirm Vercel builds production rollover PR branches. The verifier
   associates the payload with GitHub only when exactly one open automation PR
   has the expected head SHA/ref and `main` base.

The production and preview Environments, active `Protect main` ruleset, and
Vercel preview bypass configuration remain administrator-managed external
controls; their private values never belong in repository code.

## Initial My Drive cycle

The current July form may remain owner-controlled in My Drive. This is a narrow
bootstrap exception, not a second managed storage location. The code permits an
explicit fallback only for the immutable `2026-07` source cycle; a missing
managed source in any later cycle fails instead of reusing July:

1. Put the private form ID only in the production protected Environment. Do not
   put it in a repository file or ordinary GitHub variable.
2. Give the production service account a direct Forms Editor/Drive `writer`
   permission through activation, recovery, and active verification.
3. Share July with the production editor group as Editor, remove the internal
   editors' direct ACLs after verification, and keep the one external editor as
   a direct production writer.
4. The Google copy operation does not carry the source ACL into the destination.
   The destination first inherits its Shared Drive access, then the automation
   reconciles the anonymous published-reader permission and the environment's
   intended collaborators. Production copies its production group and direct
   external writer. Owner and
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
5. After the August form is active and verified, the owner may remove the direct
   service-account ACL from July. Later cycles resolve only managed forms in the
   production Shared Drive, even if the stale fallback secret remains.

If Workspace policy refuses direct sharing to a service-account principal, move
the source into the production Shared Drive or redesign the identity boundary.
Do not work around that policy with domain-wide delegation or reusable Google
credentials.

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
  target for the cycle, creates or resumes that target's distinct private
  response spreadsheet, keeps the form published but closed, and reconstructs
  the one marker-only draft page PR.
- `sync-responses` verifies that the requested source cycle is the currently
  open form named on the website, then appends only response IDs not already in
  its managed monthly spreadsheet. The operation creates the spreadsheet if
  needed and is safe to repeat. It does not require the activation reviewer,
  cannot change the website or form, and emits counts rather than response or
  spreadsheet identifiers.
- `verify` with `prepared` requires that exact page-only PR on current `main`;
  `active` requires the completed website and Google close/open state.
- `activate` should be dispatched near 23:40 Pacific on the final day. The exact
  PR tests and Vercel preview pass first. Approval is offered only when the real
  cutoff is at most one hour away (or has passed), through the secret-free
  `progress-prizes-production-activation` Environment. The job waits without a
  Google token, authenticates at cutoff, reacquires a zero-wait GitHub lease,
  closes the source, performs its final append-only response sync, opens and
  reload-verifies the target, then merges only the activated commit.
- `reconcile-active` is a manual break-glass repair for a rollover that humans
  already completed. Run `verify active` first. Reconciliation proceeds only if
  the old form is published and closed, the new form is published and open, the
  website points exactly to the new responder URL, and titles, structure,
  linked-Sheet status, capabilities, ACLs, and copy fingerprint all match. It
  requires the same secret-free activation Environment approval and may update
  only the source `CLOSED` and target `ACTIVE` Drive `appProperties`. It never
  changes a form, response, permission, folder, publishing state, or website
  file. A second run is read-only and succeeds idempotently.

If preparation, activation, or merge stops, rerun the same operation and cycle.
Managed Drive markers make Google copy and close/open recovery idempotent. A
rerun after a completed merge performs read-only active verification. If `main`
moved before mutation, activation reconstructs and rechecks a stale-parent page
commit; any multi-file, merge, wrong-path, or query-bearing change fails closed.
Never use simulated time, fault controls, alternate branches, or staging folders
for production; those inputs are absent and the shared action rejects them
before authentication.

Manual recovery uses public controls only; Google configuration stays inside the
protected Environment:

```bash
gh workflow run progress-prizes-production.yml --ref main \
  -f operation=activate -f source-cycle=2026-07 -f target-cycle=2026-08 \
  -f verify-mode=active
gh workflow run progress-prizes-production.yml --ref main \
  -f operation=verify -f source-cycle=2026-07 -f target-cycle=2026-08 \
  -f verify-mode=active
gh workflow run progress-prizes-production.yml --ref main \
  -f operation=reconcile-active -f source-cycle=2026-07 -f target-cycle=2026-08 \
  -f verify-mode=active
```

The trusted GitHub coordinator prints only these diagnostic codes:
`main-ref-moved`, `ambiguous-pr`, `invalid-page-head`,
`invalid-pr-association`, `invalid-main-ref`, `completed-state-missing`, and
`unexpected-error`. The code is safe to include in an incident report; API
bodies and private identifiers are never printed.

Generated `codex/progress-prize-YYYY-MM` branches and their PRs are strictly
marker-only. Never push a human guideline, copy, or formatting commit to a
generated rollover branch. Put editorial changes on a separate branch and PR;
otherwise the exact one-commit/one-file activation gate will reject the
rollover. Production never deletes an old form or any response. Cleanup is not
exposed by the production workflow.

## Monthly response spreadsheets

The automation uses a private, app-property-managed spreadsheet rather than
asking Google Forms to create a native response destination. The Forms API
exposes `linkedSheetId` only as output, and a headless WIF service account cannot
use the interactive Forms UI picker. This keeps authentication keyless while
retaining Forms as the authoritative response record.

One spreadsheet is created in the same environment's active Shared Drive folder
for each form cycle. It is fingerprint-bound to that exact form and receives
only the configured direct internal editor collaborators plus the inherited
Shared Drive Managers. It never receives the anonymous responder permission.
The first row is a stable header; every later row contains the response ID,
timestamps, respondent email when Google supplies it, answers aligned by
question ID, and a raw-answer JSON fallback. Appends use `RAW` input mode so a
response beginning with `=` is stored as data, not evaluated as a formula.

The append protocol first scans column A for existing response IDs, obtains all
current Form responses through the read-only Forms Responses API, and appends
only unseen IDs. After an ambiguous network failure it scans the IDs again
before retrying, preventing a duplicate append. A later edit to a Form response
does not rewrite its existing spreadsheet row; the latest canonical value
remains available in Google Forms. There are intentionally no code paths for
Sheets value update, clear, row deletion, spreadsheet deletion, or production
file movement.

Existing native linked Sheets are legacy records. Their IDs are treated as
private, and their cells, ACLs, filenames, folders, and Form linkage are never
mutated. A managed append-only spreadsheet may coexist with a legacy linked
Sheet for the initial cycle. Old forms and all production monthly spreadsheets
stay in place for response review after rollover.

To backfill or check the currently active month without waiting for the daily
schedule, dispatch only public cycle controls; no Google identifier is supplied
on the command line:

```bash
gh workflow run progress-prizes-production.yml --ref main \
  -f operation=sync-responses -f source-cycle=2026-08 -f target-cycle=2026-09 \
  -f verify-mode=prepared
```

## Automated schedule and immediate verification

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
- `07:27` Pacific is the daily append-only response sync for the currently
  active Pacific month. It is suppressed while another production rollover job
  is nonterminal and fails closed if the form, website, or managed spreadsheet
  binding is inconsistent.
- `23:40` Pacific on candidate final days dispatches activation only when the
  observed date is the actual last calendar day. The production workflow—not
  the scheduler—runs tests and the Vercel gate, requests human approval, waits
  for cutoff without a Google token, then authenticates.
- `00:17` Pacific remains a first-day recovery probe. `06:47` Pacific retries
  on days 1–7, always targeting the previous month's incomplete rollover. A
  delayed day-7 event may cross local midnight before the day-8 slot; a true
  day-8 event no-ops. If an exact production run is already nonterminal, the
  scheduler does not enqueue a stale duplicate. A completed rollover reaches
  read-only active verification and never creates another form or repeats
  publishing mutations.

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

The workflow contract test checks immutable action pins, OIDC isolation,
repository IDs, retired staging controls, exact-commit Vercel
association, and trusted GitHub check provenance. The gated production and
scheduler have separate executable contract tests for the clock boundaries,
minimal permissions, deduplication, fixed dispatch endpoint, child-run binding,
redacted failures, and absence of Google configuration.
