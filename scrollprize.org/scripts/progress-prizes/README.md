# Progress Prize form rollover

This automation prepares the next monthly Google Form, changes only the managed
deadline and responder link in `docs/34_prizes.md`, opens that change as a
draft pull request, and marks the pull request ready once the month's deadline
has passed and its checks are green. The cutoff itself is manual: a human
closes the old form, opens the new one, and merges the pull request. The
cutoff is midnight immediately after the last calendar day in
`America/Los_Angeles` (the published deadline remains 11:59pm Pacific).

Responses live only in Google Forms; the automation never reads or copies them.
Anyone who wants a spreadsheet can use **Link to Sheets** in a form's Responses
tab. A linked Sheet is ignored by every check, and the automation never reads
its cells, changes its link, moves it, or deletes it.

A fresh copy is closed at the first possible Forms API call, before title, ACL,
or capability reconciliation, so a prepared form never accepts responses early.
The source form only has to stay published. If someone closes it by hand at or
before the deadline, `prepare` and `verify` still proceed.

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
  `prepare`, `ready`, and `verify` operations. Every authenticated job is
  literal in that top-level workflow; `ready` touches only GitHub.
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

Create these environments before merging the workflows. Restrict both to the
protected `main` branch. Neither may require a reviewer: a queued daily
preparation must never hold the production concurrency lock, and the human
decision point is the manual merge of the rollover pull request, not an
Environment approval.

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
renamed. The August copy is created in the production Shared Drive and is
cryptographically bound to that exact source ID. From then on, managed
`appProperties` discovery always selects the prior managed target, so the
fallback secret does not need a monthly edit and is ignored for later cycles.

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

Enable the Google Forms and Drive APIs. Create the production service
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
`validate`, `verify`, and dry runs use read-only Forms body and Drive scopes.
The form copy uses `forms.body` and `drive`: this headless workflow must find
pre-existing forms and app-property-managed files in a Shared Drive, which
`drive.file` cannot reliably authorize without an interactive picker. No
response or Sheets scope is requested. The separate service accounts, the two
exact-file ACLs on the initial My Drive form, and the isolated Shared Drive ACLs
bound the writable resources.
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
   requests**. Without this, draft creation and the draft-to-ready transition
   fail closed. GitHub places `pull_request` checks triggered by a PR created with
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
   permission until the August copy is prepared and verified.
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
5. After the August form is prepared and verified, the owner may remove the
   direct service-account ACL from July. Later cycles resolve only managed forms in the
   production Shared Drive, even if the stale fallback secret remains.

If Workspace policy refuses direct sharing to a service-account principal, move
the source into the production Shared Drive or redesign the identity boundary.
Do not work around that policy with domain-wide delegation or reusable Google
credentials.

## Production operations and recovery

Use **Progress Prize production rollover** on `main`. The source cycle is the
month that is ending and the target cycle must always be the immediately
following month. Leave `request-id` empty for every manual run.

- `validate` performs the read-only live-form, capability, publishing, ACL, and
  copy preflight. It never writes Google or GitHub state.
- `dry-run` is deliberately useful before the normal seven-day preparation
  window. It uses the real clock and the same preparation preflight, but extends
  only the read-only planning window to 31 days. It creates no copy, changes no
  ACL or publishing state, writes no website branch, and records only the
  proposed public title and deadline in the run summary. Real `prepare` remains
  fixed to seven days.
- `prepare` is safe to dispatch repeatedly. It succeeds without opening a page
  PR outside the seven-day window. Inside the window it resumes the one managed
  target for the cycle, keeps the new form published but closed, and
  reconstructs the one marker-only draft page PR. The source may be open or
  already closed.
- `ready` is the cutoff step. It finds the one open rollover PR, checks that its
  head is a single page-only commit with the exact deterministic deadline and
  link change, waits up to 30 minutes for the `Public no-secret tests` check and
  the exact Vercel preview status on that commit, then marks the draft ready for
  review and posts the manual cutoff checklist as a PR comment. It never
  merges, never authenticates to Google, and other repository workflows on the
  same commit cannot block it. If the PR is already ready it changes nothing;
  if the page is already on `main` it exits green.
- `verify` is read-only. It requires the one open rollover PR and checks that
  the prepared form and the PR's page agree on cycle, link, titles, structure,
  capabilities, ACLs, and copy fingerprint.

The manual cutoff, once the PR is ready:

1. In Google Forms, stop the previous month's form accepting responses.
2. Open the new month's form; it is already published and closed.
3. Merge the pull request (squash). Order relative to the form switch does not
   matter to the automation.

If preparation stops, rerun `prepare` for the same cycles; managed Drive
markers make the copy idempotent. If `ready` stops, rerun it; it is idempotent
too. Manual recovery uses public controls only:

```bash
gh workflow run progress-prizes-production.yml --ref main \
  -f operation=prepare -f source-cycle=2026-11 -f target-cycle=2026-12
gh workflow run progress-prizes-production.yml --ref main \
  -f operation=ready -f source-cycle=2026-11 -f target-cycle=2026-12
gh workflow run progress-prizes-production.yml --ref main \
  -f operation=verify -f source-cycle=2026-11 -f target-cycle=2026-12
```

The trusted GitHub coordinator prints only these diagnostic codes:
`ambiguous-pr`, `invalid-pr-association`, `invalid-main-ref`,
`completed-state-missing`, and `unexpected-error`. The code is safe to include
in an incident report; API bodies and private identifiers are never printed.

Generated `codex/progress-prize-YYYY-MM` branches and their PRs are strictly
marker-only. Never push a human guideline, copy, or formatting commit to a
generated rollover branch. Put editorial changes on a separate branch and PR;
otherwise the one-commit/one-file check will refuse to mark the rollover ready.
Production never deletes an old form or any response.

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
- `23:40` Pacific on candidate final days dispatches `ready` only when the
  observed date is the actual last calendar day. The production workflow—not
  the scheduler—waits for the tests and the Vercel preview and flips the draft.
- `00:17` Pacific remains a first-day retry. `06:47` Pacific retries on days
  1–7, always targeting the previous month's rollover. A delayed day-7 event
  may cross local midnight before the day-8 slot; a true day-8 event no-ops. If
  an exact production run is already nonterminal, the scheduler does not
  enqueue a stale duplicate. A PR that is already ready, or already merged,
  makes every retry a green no-op.

Production and scheduler concurrency groups never cancel an in-progress run and
use GitHub's queued concurrency mode so a manual race cannot silently replace a
pending cutoff run. The scheduler itself never sleeps. GitHub may delay or drop
a scheduled event, and public-repository schedules can be disabled after 60 days
without repository activity, so the production workflow keeps its manual
`prepare` and `ready` recovery path.

Immediately after the scheduler reaches `main`, manually dispatch **Progress
Prize production schedule** once. Manual scheduler runs have no inputs and can
only dispatch the real-clock production `dry-run`; they cannot prepare, mark
ready, close, open, share, publish, or update the website. Verify that the
scheduler's recorded child run is the exact successful read-only production run.

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
repository IDs, the absence of retired staging and activation controls,
exact-commit Vercel association, and trusted GitHub check provenance. The gated production and
scheduler have separate executable contract tests for the clock boundaries,
minimal permissions, deduplication, fixed dispatch endpoint, child-run binding,
redacted failures, and absence of Google configuration.
