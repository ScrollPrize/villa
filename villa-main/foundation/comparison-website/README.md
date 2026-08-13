# Pairwise Image Preference Collector

Minimal FastAPI + Nginx app used to collect pairwise labels from users.

The system stores preferences in `data/preferences.db` (SQLite) and serves images
from `./images` to the frontend.

## Data model and logging

- `left` / `right` preferences are stored as `left` or `right` only.
- Each logged row stores:
  - `pair_id`
  - `user_id`
  - `fold`
  - `sample`
  - `left_image`
  - `right_image`
  - `preference`
- DB path: `./data/preferences.db` (mounted into the `api` container at
  `/app/data/preferences.db`).

## API

- `GET /api/catalog` returns available folds and samples.
- `GET /api/pairs?fold=<fold>&sample=<sample>` returns one random pair.
- `POST /api/preferences` stores preference payload.
- `GET /api/preferences?limit=<n>` lists latest rows.
- `GET /api/health` health check.

## Compose services

`docker-compose.yml` exposes:

- `api` (FastAPI)
- `web` (nginx on port `8080`)

Image root mount:
- active dataset: `./images` -> `/data/images`
- legacy dataset (optional): `./images_deprecated` -> `/data/images_deprecated`

## Dataset migration plan

Use the legacy folder to preserve old images while switching to the new
component-paired dataset.

### 1) Archive current images

```bash
mkdir -p images_deprecated
mv images images_deprecated/current_legacy
mkdir -p images
```

### 2) Generate component dataset from the new source

```bash
python3 scripts/build_component_pairs_dataset.py \
  --source /home/giorgio/scrolls/<your_dataset_root> \
  --output images \
  --min-area 64 \
  --score-threshold 0.05 \
  --margin 8 \
  --crop-mode components \
  --jobs 32
```

If you want to use a fast split-based generation for scroll-4 style sources, use the four-way width split (four vertical stripes):

```bash
python3 scripts/build_component_pairs_dataset.py \
  --source /home/giorgio/scrolls/s4_archive_images \
  --output /tmp/s4_four_pairs \
  --allow-scroll4 \
  --crop-mode four \
  --jobs 32
```

For legacy behavior you can still run split mode in halves:

```bash
python3 scripts/build_component_pairs_dataset.py \
  --source /home/giorgio/scrolls/s4_archive_images \
  --output /tmp/s4_half_pairs \
  --allow-scroll4 \
  --crop-mode half \
  --jobs 32
```

The generator refuses to run on Scroll-4 dataset roots by default. If you really need it, pass:

```bash
--allow-scroll4
```

Optional flags:
- `--overwrite` replace an existing processed image/manifest output.
- `--dry-run` validate pair generation only.
- `--source` and `--output` are required.

The script produces:
- component crops in `images/<fold>/<sample>/<pair_id>_left.png` and
  `images/<fold>/<sample>/<pair_id>_right.png`
- per-sample manifest `images/<fold>/<sample>/pairs.jsonl`
- global manifest `data/component_pairs_index.jsonl`

### 3) Deploy updated site

```bash
docker compose down
docker compose up --build -d
```

### 4) Validate

```bash
curl http://localhost:8080/api/catalog
curl "http://localhost:8080/api/pairs?fold=l_2&sample=line_04"
curl http://localhost:8080/api/preferences?limit=5
```

## Data persistence note

`data/preferences.db` is untouched by migration.
- Existing rows continue to exist after moving images.
- New labels written after migration use rows for new image pairs (new IDs) while the
  old rows remain valid for audit.
- If the containers are stopped and restarted, DB data remains as long as
  `./data` is kept intact in your working directory.

## Legacy inspection

Archived source corpus is available at:
- `./images_deprecated/current_legacy`

You can inspect old rows from the DB at any time with:

```bash
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect('data/preferences.db')
cur = conn.execute('SELECT id, fold, sample, left_image, right_image, pair_id FROM preference_logs ORDER BY id DESC LIMIT 10')
for row in cur:
    print(row)
PY
```

## Quick start

```bash
cp .env.example .env # optional

docker compose up --build -d
```

Open:
- UI: http://localhost:8080
- API catalog: http://localhost:8080/api/catalog
- API health: http://localhost:8080/api/health
- API pair: http://localhost:8080/api/pairs
