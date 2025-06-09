#!/usr/bin/env bash
set -e

echo "[ENTRYPOINT] Fetching AWS credentials…"
creds=$(
  aws sts get-caller-identity \
    --profile $PROFILE >/dev/null \
  && aws configure export-credentials \
       --profile $PROFILE
)

echo "[ENTRYPOINT] Exporting into environment…"
export AWS_ACCESS_KEY_ID=$(echo "$creds" | jq -r .AccessKeyId)
export AWS_SECRET_ACCESS_KEY=$(echo "$creds" | jq -r .SecretAccessKey)
export AWS_SESSION_TOKEN=$(echo "$creds" | jq -r .SessionToken)

echo "[ENTRYPOINT] Mounting S3 bucket with goofys…"
mkdir -p /mnt/"$BUCKET"
# Mount in background so we can run the Qt app afterwards
goofys $BUCKET /mnt/"$BUCKET"

# (Optional) wait a second to let the mount settle:
sleep 1

echo "[ENTRYPOINT] Dropping into bash shell"
exec /bin/bash