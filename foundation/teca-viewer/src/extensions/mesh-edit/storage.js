import { transaction } from "../../storage.js";

const STORES = ["edits", "surfaces"];
const DATABASE = {
  name: "teca-mesh-edit",
  version: 1,
  upgrade: (database) => {
    for (const name of STORES) database.createObjectStore(name);
  },
};

function editsFrom(volumeUrl, surfaceUrl, seq) {
  return IDBKeyRange.bound([volumeUrl, surfaceUrl, seq], [volumeUrl, surfaceUrl, Number.POSITIVE_INFINITY]);
}

export async function readLog(volumeUrl, surfaceUrl) {
  const [edits, surface] = await transaction(DATABASE, STORES, "readonly", (edits, surfaces) => [
    edits.getAll(editsFrom(volumeUrl, surfaceUrl, 0)),
    surfaces.get([volumeUrl, surfaceUrl]),
  ]);
  return { edits, done: surface?.done ?? 0 };
}

export async function writeLog(volumeUrl, surfaceUrl, done, edits, from) {
  await transaction(DATABASE, STORES, "readwrite", (store, surfaces) => {
    if (edits.length > 0) {
      store.delete(editsFrom(volumeUrl, surfaceUrl, from));
      for (const [position, edit] of edits.entries()) store.put(edit, [volumeUrl, surfaceUrl, from + position]);
    }
    surfaces.put({ done }, [volumeUrl, surfaceUrl]);
  });
}

export async function forgetLog(volumeUrl, surfaceUrl) {
  await transaction(DATABASE, STORES, "readwrite", (edits, surfaces) => {
    edits.delete(editsFrom(volumeUrl, surfaceUrl, 0));
    surfaces.delete([volumeUrl, surfaceUrl]);
  });
}
