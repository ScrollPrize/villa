const CATALOG_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json";
const BUCKET = "s3://vesuvius-challenge-open-data";
const BUCKET_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";

function assetUrl(asset) {
  for (const origin of asset.origins)
    if (origin.access_roots.some((root) => root.url === BUCKET)) return `${BUCKET_URL}/${origin.path}`;
  for (const origin of asset.origins)
    for (const root of origin.access_roots) if (root.type === "https") return `${root.url}/${origin.path}`;
  return null;
}

export function catalogInk(metadata) {
  const images = new Map();
  for (const sample of Object.values(metadata.samples))
    for (const segment of Object.values(sample.segments)) {
      const typed = (type) => segment.data.filter((asset) => asset.type === type && assetUrl(asset) !== null);
      for (const surface of typed("tifxyz-transformed")) {
        const found = typed("ink-detection-downsampled")
          .filter((image) => image.parameters.target_volume === surface.parameters.target_volume)
          .map((image) => {
            const { target_volume, model_id, render_level } = image.parameters;
            const model = metadata.models[model_id];
            if (model === undefined) throw new Error(`The catalog lists no model ${model_id}, of ${assetUrl(image)}.`);
            const tiff = typed("ink-detection").find(
              ({ parameters }) =>
                parameters.target_volume === target_volume &&
                parameters.model_id === model_id &&
                parameters.render_level === render_level,
            );
            return { name: model.long_id, image: assetUrl(image), tiff: tiff === undefined ? null : assetUrl(tiff) };
          });
        if (found.length > 0) images.set(assetUrl(surface).replace(/\/+$/, ""), found);
      }
    }
  return images;
}

export class CatalogChoice {
  constructor(select, surface, picked, failed) {
    this.select = select;
    this.surface = surface;
    this.failed = failed;
    this.images = null;
    this.reading = false;
    select.addEventListener("pointerdown", () => this.read());
    select.addEventListener("keydown", () => this.read());
    select.addEventListener("change", () => {
      picked(this.images.get(surface()).find((candidate) => candidate.image === select.value));
      select.value = "";
    });
    this.offer();
  }

  offer() {
    const found = this.images?.get(this.surface()) ?? [];
    const first = this.reading
      ? "Reading the catalog…"
      : this.images === null
        ? "The catalog's ink images"
        : found.length === 0
          ? "The catalog has none for this surface"
          : `The catalog's ink images (${found.length})`;
    this.select.replaceChildren(new Option(first, ""), ...found.map(({ name, image }) => new Option(name, image)));
  }

  async read() {
    if (this.images !== null || this.reading) return;
    this.reading = true;
    this.offer();
    try {
      const response = await fetch(CATALOG_URL, { cache: "no-cache" });
      if (!response.ok) throw new Error(`The catalog ${CATALOG_URL} answered ${response.status}.`);
      this.images = catalogInk(await response.json());
    } catch (error) {
      this.failed(error);
    } finally {
      this.reading = false;
      this.offer();
    }
  }
}
