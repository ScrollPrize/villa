#!/usr/bin/env bash
# Sign the VC3D.app in a DMG that CPack made with a Developer ID, notarize and staple it, and rebuild the DMG and its
# ZIP in place under the same names, with the DMG signed, notarized and stapled too. The app then passes Gatekeeper
# whether it is opened from the DMG or after it is copied out, online or not. CPack's ad-hoc signed app is the input;
# without the secrets the release workflow skips this script and publishes that one unchanged
#   sign-macos.sh DMG ZIP
# Environment: SIGN_IDENTITY, a Developer ID Application identity in the keychain; NOTARY_KEY_PATH (an App Store
# Connect API key, .p8), NOTARY_KEY_ID and NOTARY_ISSUER_ID
set -euo pipefail
dmg=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); zip=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
: "${SIGN_IDENTITY:?set SIGN_IDENTITY to a Developer ID Application identity}"
: "${NOTARY_KEY_PATH:?set NOTARY_KEY_PATH}" "${NOTARY_KEY_ID:?set NOTARY_KEY_ID}" "${NOTARY_ISSUER_ID:?set NOTARY_ISSUER_ID}"
work=$(mktemp -d)
mnt=$work/mnt
trap 'hdiutil detach -quiet "$mnt" 2>/dev/null || true; rm -rf "$work"' EXIT

notarize() {
  local log status id
  log=$work/notary-$(basename "$1").json
  xcrun notarytool submit "$1" --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID" \
    --wait --timeout 1h --output-format json > "$log" || true
  # the status decides, so a rejected submission still gets its log printed
  status=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "$log")
  id=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["id"])' "$log")
  echo "notarization of $(basename "$1"): $status ($id)"
  if [ "$status" != Accepted ]; then
    xcrun notarytool log "$id" --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID" || true
    exit 1
  fi
}

sign() {
  codesign --force --timestamp --options runtime --sign "$SIGN_IDENTITY" "$@"
}

echo "== unpack $dmg at $(date)"
mkdir -p "$mnt"
hdiutil attach -quiet -nobrowse -readonly -mountpoint "$mnt" "$dmg"
volname=$(diskutil info "$mnt" | awk -F': *' '/Volume Name/ {print $2; exit}')
# the DMG's layout as CPack made it (the app, the Applications link, anything else), with the app replaced below
stage=$work/stage
ditto "$mnt" "$stage"
hdiutil detach -quiet "$mnt"
app=$(find "$stage" -maxdepth 1 -name '*.app' | head -1)
[ -d "$app" ] || { echo "no app in $dmg" >&2; exit 1; }
echo "volume $volname, app $(basename "$app")"

echo "== sign $(basename "$app") with $SIGN_IDENTITY at $(date)"
# inside out: loose Mach-O files (dylibs, Qt plugins, the vc_* tools) deepest first, then each framework bundle, then
# the app, whose main executable is signed with it
main=$app/Contents/MacOS/$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$app/Contents/Info.plist")
while IFS= read -r f; do
  # a match on file's output, not a pipe into grep -q: under pipefail a SIGPIPE there would skip the file unsigned
  if [[ $(file -b "$f") == *Mach-O* ]]; then sign "$f"; fi
done < <(find "$app/Contents" -type f -not -path '*.framework/*' -not -path "$main" \
  | awk -F/ '{print NF "\t" $0}' | sort -rn | cut -f2-)
find "$app/Contents/Frameworks" -maxdepth 1 -name '*.framework' -print0 | while IFS= read -r -d '' framework; do
  sign "$framework"
done
sign "$app"
codesign --verify --deep --strict --verbose=2 "$app"

echo "== notarize the app at $(date)"
ditto -c -k --keepParent "$app" "$work/app.zip"
notarize "$work/app.zip"
xcrun stapler staple "$app"

echo "== DMG and ZIP at $(date)"
rm -f "$dmg" "$zip"
hdiutil create -quiet -volname "$volname" -srcfolder "$stage" -format UDZO "$dmg"
codesign --force --timestamp --sign "$SIGN_IDENTITY" "$dmg"
notarize "$dmg"
xcrun stapler staple "$dmg"
ditto -c -k --sequesterRsrc --keepParent "$app" "$zip"

echo "== checks"
spctl -a -vv "$app"
spctl -a -t open --context context:primary-signature -vv "$dmg"
xcrun stapler validate "$app"
xcrun stapler validate "$dmg"
hdiutil attach -quiet -nobrowse -readonly -mountpoint "$mnt" "$dmg"
ls -la "$mnt"
spctl -a -vv "$mnt/$(basename "$app")"
hdiutil detach -quiet "$mnt"
shasum -a 256 "$dmg" "$zip"
echo "signed and notarized $dmg and $zip at $(date)"
