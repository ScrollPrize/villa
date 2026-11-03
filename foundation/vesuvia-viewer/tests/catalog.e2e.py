import json
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
HASH = "#ext=extensions/catalog.js?index=outputs/e2e-catalog.json"


def origin(path):
    return {"origins": [{"path": path, "access_roots": [{"type": "s3", "url": "s3://fixture"}]}]}


OUTPUTS.mkdir(exist_ok=True)
(OUTPUTS / "e2e-catalog.json").write_text(json.dumps({
    "metadataUrl": "e2e-catalog-metadata.json",
    "dataAccess": {"rewrites": [{"from": "s3://fixture/", "to": "https://data.example/"}]},
}))
(OUTPUTS / "e2e-catalog-metadata.json").write_text(json.dumps({"samples": {
    "Scroll1": {
        "volumes": {"900": {"properties": {"pixel_size_um": 2.4}, "data": [
            {"type": "ome-zarr", **origin("Scroll1/volumes/900-2.4um.zarr/")},
            {"type": "surface-prediction-zarr", **origin("Scroll1/predictions/900-surface.zarr/")},
        ]}},
        "segments": {"s1": {"data": [
            {"type": "tifxyz-transformed", "parameters": {"target_volume": "900"},
             **origin("Scroll1/segments/s1/mesh/s1-on-900-2.4um.tifxyz/")},
        ]}},
    },
    "Scroll2": {
        "volumes": {"901": {"properties": {"pixel_size_um": 8}, "data": [{"type": "ome-zarr", **origin("Scroll2/volumes/901-8um.zarr/")}]}},
        "segments": {},
    },
}}))


async def check(browser):
    await browser.wait_for("document.getElementById('catalog-note')?.textContent.includes('volumes')", 30)
    catalog = await browser.evaluate("""(() => {
      const load = vesuvia.load, opened = [];
      vesuvia.load = state => { opened.push(state); return Promise.resolve() };
      const volumes = document.getElementById('catalog-volume'), surfaces = document.getElementById('catalog-surface');
      const listed = [...volumes.children].map(group => [group.label, [...group.children].map(option => option.textContent)]);
      volumes.value = '1'; volumes.dispatchEvent(new Event('change'));
      const bare = { surfaces: surfaces.options.length, predictions: document.getElementById('catalog-layer-0') === null };
      volumes.value = '0'; volumes.dispatchEvent(new Event('change'));
      surfaces.value = '0';
      document.getElementById('catalog-open').click();
      vesuvia.load = load;
      return { listed, bare, opened, note: document.getElementById('catalog-note').textContent,
        surfaces: [...surfaces.options].map(option => option.textContent),
        header: document.getElementById('extensions-loaded').textContent };
    })()""")
    if (catalog["listed"] != [["Scroll1", ["2.4 µm, 900"]], ["Scroll2", ["8 µm, 901"]]]
            or catalog["bare"] != {"surfaces": 1, "predictions": True}
            or catalog["surfaces"] != ["none", "s1-on-900-2.4um.tifxyz"]
            or "Opening 900-2.4um.zarr" not in catalog["note"]
            or not catalog["header"].startswith("ext: catalog.js")
            or catalog["opened"] != [{"ct": "https://data.example/Scroll1/volumes/900-2.4um.zarr/",
                                      "surface": "https://data.example/Scroll1/segments/s1/mesh/s1-on-900-2.4um.tifxyz/",
                                      "layers": ["https://data.example/Scroll1/predictions/900-surface.zarr/"]}]):
        return [f"{catalog}"]
    return []
