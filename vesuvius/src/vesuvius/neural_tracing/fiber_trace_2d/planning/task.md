# Task: minimize manager staging and Atlas inference ingestion

Remove the manager-specific `upload-manifest.json` protocol. S3 staging must
use rclone's normal resumable copy behavior with only `_INCOMPLETE` as the
publication guard: create the marker before transfer and remove it only after
rclone succeeds.

Keep portable `inference.json` and all previously approved model metadata.
Do not copy portable inference provenance into an Atlas data entry's
`creation_info`. Reuse the existing Atlas `lasagna` copy-first representation
without extending its data-entry schema: store only the private origin and the
existing `model_id` and `level` parameters.

Clean the already staged/ingested work-in-progress runs consistently, without
changing their artifacts, model identities, origins, parameters, or public
publication state.
