# Pairwise Image Preference Collector

Minimal full-stack setup to collect pairwise preference labels for images.

## Features
- Serves images from an existing folder via Nginx at `/images/...`
- Generates random image pairs from:
  `./images/<fold>/<sample>/<images>`
- Collects user preference as: `left` or `right`
- Logs each response into a SQLite DB (`preferences.db`) stored under `./data`.
- Docker Compose deployment with:
  - `api` (FastAPI + SQLite)
  - `web` (Nginx static + API reverse proxy)

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

Database file is stored at `./data/preferences.db` on the host (mounted into the container at `/app/data/preferences.db`).

This setup makes preference data persistent even across container restarts, recreates, and image updates, as long as you keep the project directory.

If you previously used the old Docker named volume `prefs-data`, migrate once with:

```bash
mkdir -p data
docker compose stop api
docker run --rm -v comparison-website_prefs-data:/from -v "$PWD/data:/to" alpine:3.20 sh -lc 'cp -a /from/preferences.db /to/preferences.db'
docker compose start api
```
