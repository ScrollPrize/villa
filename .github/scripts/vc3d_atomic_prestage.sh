#!/usr/bin/env bash
# The two tags reference the SAME immutable annotated tag object.
# PRE-STAGE while the source commit is current; do NOT mint old tags later.
set -euo pipefail
channel="${1:?channel}"; candidate="${2:?candidate}"
old_oid="${3:?old_oid}"; previous_archive="${4:?previous_archive}"
previous_id="${5:?previous_id}"
asset_ids="${6:?six-original-asset-IDs}"
[[ "$channel" == latest || "$channel" == stable ]] || exit 2
[[ "$candidate" =~ ^[a-f0-9]{40}$ && "$old_oid" =~ ^[a-f0-9]{40}$ ]] || exit 2
[[ "$previous_archive" =~ ^vc3d-archive-${channel}-[0-9a-f]{12}-[1-9][0-9]*$ ]] || exit 2
[[ "$previous_id" =~ ^[1-9][0-9]*$ && "${GITHUB_RUN_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$asset_ids" =~ ^[1-9][0-9]*(,[1-9][0-9]*){5}$ ]] || exit 2
# Re-read the archived release after PATCH and BEFORE writing an enduring
# receipt; a relisted/reuploaded asset would gain a different numeric ID.
seen_ids="$(gh api "repos/$GITHUB_REPOSITORY/releases/$previous_id" | python3 -c   'import sys,json; d=json.load(sys.stdin); print(",".join(str(i) for i in sorted(a["id"] for a in d["assets"])))')"
[[ "$seen_ids" == "$asset_ids" ]] || { echo "::error::archived IDs changed before prestage" >&2; exit 7; }
stage="vc3d-archive-${channel}-${candidate:0:12}-${GITHUB_RUN_ID}"
if git ls-remote --exit-code --refs origin "refs/tags/$stage" >/dev/null 2>&1; then
  echo "::error::archive alias already exists: $stage" >&2; exit 3
fi
# A source mismatch must not be able to mint an apparently valid archive.
git rev-parse --verify "${candidate}^{commit}" >/dev/null
# Do not let two different checkout refs create conflicting annotations.
if [[ "$channel" == latest ]]; then title="VC3D latest release at $candidate";
else title="VC3D stable release at $candidate"; fi
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git tag --force --annotate "$channel" "$candidate" \
  --message "$title" \
  --message "Previous archive: $previous_archive" \
  --message "Previous archive release ID: $previous_id" \
  --message "Previous archive tag object: $old_oid" \
  --message "Previous archive asset IDs: $asset_ids" \
  --message "Staged archive: $stage"
oid="$(git rev-parse "refs/tags/$channel")"
[[ "$(git cat-file -t "$oid")" == tag &&
   "$(git rev-parse "${oid}^{commit}")" == "$candidate" ]] || exit 4
# The archive release is already retargeted to this alias. Reject a moved or
# missing alias before promoting the channel, even when its rolling ref stayed.
archive_oid="$(git ls-remote --refs origin "refs/tags/$previous_archive" | cut -f1)"
if [[ "$archive_oid" != "$old_oid" ]]; then
  echo "::error::historical archive alias lost exact tag object: $previous_archive" >&2
  exit 6
fi
# Atomic and CAS: if EITHER remote ref cannot be safely updated, NEITHER changes.
git push --atomic \
  --force-with-lease="refs/tags/$channel:$old_oid" \
  --force-with-lease="refs/tags/$stage:" \
  origin "refs/tags/$channel:refs/tags/$channel" \
  "refs/tags/$channel:refs/tags/$stage"
# Readback: preserve the exact ANNOTATED tag object, not only peeled commit.
[[ "$(git ls-remote --refs origin "refs/tags/$channel" | cut -f1)" == "$oid" ]] || exit 5
[[ "$(git ls-remote --refs origin "refs/tags/$stage" | cut -f1)" == "$oid" ]] || exit 5
# Fail visibly if an external actor moved the historical alias during push.
[[ "$(git ls-remote --refs origin "refs/tags/$previous_archive" | cut -f1)" == "$old_oid" ]] || {
  echo "::error::historical archive alias raced during tag promotion" >&2
  exit 6
}
printf 'P116_ATOMIC_PRESTAGE_OK channel=%s tag_oid=%s stage=%s commit=%s\n' \
  "$channel" "$oid" "$stage" "$candidate"
