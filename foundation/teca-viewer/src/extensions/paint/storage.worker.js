import { transaction } from "../../storage.js";
import { through } from "./file.js";
import { CHUNKS, chunkRange, DATABASE, EDITS, HEADS, JOURNAL, NONE, numberRange } from "./storage.js";

async function read(url, part, parts) {
  const kept = part === 0;
  const [paths, packed, keys, entries, ids, edits, head] = await transaction(
    DATABASE,
    kept ? [CHUNKS, JOURNAL, EDITS, HEADS] : [CHUNKS],
    "readonly",
    (store, journal, edited, heads) => [
      store.getAllKeys(chunkRange(url)),
      store.getAll(chunkRange(url)),
      ...(kept
        ? [
            journal.getAllKeys(numberRange(url)),
            journal.getAll(numberRange(url)),
            edited.getAllKeys(numberRange(url)),
            edited.getAll(numberRange(url)),
            heads.get(url),
          ]
        : []),
    ],
  );
  if (kept) {
    const arrays = edits.flatMap((edit) => [edit.chunks_N3, edit.offsets_N, edit.lengths_N, edit.values_N]);
    self.postMessage(
      {
        kept: {
          entries: keys.map(([, sequence], index) => ({ sequence, entry: entries[index] })),
          edits: ids.map(([, id], index) => [id, edits[index]]),
          head: head ?? NONE,
        },
      },
      [...new Set(arrays.map(({ buffer }) => buffer))],
    );
  }
  for (let index = part; index < paths.length; index += parts) {
    const bytes = packed[index];
    const data_DHW = await through(bytes, new DecompressionStream("gzip"));
    self.postMessage({ chunk: [index, paths[index][1], data_DHW, bytes] }, [data_DHW.buffer, bytes.buffer]);
  }
}

self.onmessage = async ({ data }) => {
  try {
    if (data.read !== undefined) await read(...data.read);
    else if (data.journal !== undefined) {
      const [url, sequence, entry, head, edit] = data.journal;
      await transaction(DATABASE, [JOURNAL, HEADS, EDITS], "readwrite", (journal, heads, edits) => {
        journal.put(entry, [url, sequence]);
        heads.put(head, url);
        if (edit === null) return;
        // Its arrays view the page's own, which may be longer, so each is kept as far as it views.
        const [chunks_N3, offsets_N, lengths_N, values_N] = [
          edit.chunks_N3,
          edit.offsets_N,
          edit.lengths_N,
          edit.values_N,
        ].map((array) => array.slice());
        edits.put({ value: edit.value, chunks_N3, offsets_N, lengths_N, values_N, bytes: edit.bytes }, [url, edit.id]);
      });
    } else if (data.forget !== undefined)
      await transaction(DATABASE, [HEADS], "readwrite", (heads) => void heads.put(NONE, data.forget));
    else if (data.compact !== undefined) {
      const [url, chunks, last, forgotten] = data.compact;
      await transaction(DATABASE, [CHUNKS, JOURNAL, EDITS], "readwrite", (store, journal, edits) => {
        for (const [path, bytes] of chunks) {
          if (bytes === null) store.delete([url, path]);
          else store.put(bytes, [url, path]);
        }
        journal.delete(numberRange(url, last));
        for (const id of forgotten) edits.delete([url, id]);
      });
    } else if (data.replace !== undefined) {
      const [url, chunks] = data.replace;
      await transaction(DATABASE, [CHUNKS, JOURNAL, EDITS, HEADS], "readwrite", (store, journal, edits, heads) => {
        store.delete(chunkRange(url));
        journal.delete(numberRange(url));
        edits.delete(numberRange(url));
        heads.delete(url);
        for (const [path, bytes] of chunks) store.put(bytes, [url, path]);
      });
    }
    self.postMessage({ id: data.id });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
};
