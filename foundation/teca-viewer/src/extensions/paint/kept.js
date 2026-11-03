import { empty, largestLabel, pathChunk, through } from "./file.js";
import { piecesOf, setTo } from "./history.js";

export class Kept {
  constructor(labels, keeping) {
    this.owner = labels;
    this.keeping = keeping;
    this.unkept = new Set();
    this.packed = new Map();
    this.sequence = 0;
    this.next = 0;
  }

  newId() {
    this.next += 1;
    return this.next - 1;
  }

  written(path) {
    this.unkept.add(path);
    this.packed.delete(path);
  }

  journal(edit, back, made) {
    const { dropped, url, history } = this.owner;
    if (dropped) return;
    this.keeping.journal(url, this.sequence, { edit: edit.id, back }, history.head(), made ? edit : null);
    this.sequence += 1;
  }

  forget() {
    this.keeping.forget(this.owner.url);
  }

  async packedChunk(path) {
    if (!this.packed.has(path)) {
      const { level, chunk_ZYX } = pathChunk(path);
      const data_DHW = this.owner.volume.chunk(level, chunk_ZYX);
      this.packed.set(path, empty(data_DHW) ? null : await through(data_DHW, new CompressionStream("gzip")));
    }
    return this.packed.get(path);
  }

  replaced(packed) {
    this.unkept.clear();
    this.packed = packed;
  }

  async restore(chunks, entries, edits, head, pause) {
    const labels = this.owner;
    let largest = 0;
    for (const [path, { data_DHW }] of chunks) {
      if (!labels.shown) return null;
      labels.put(path, data_DHW);
      if (pathChunk(path).level === 0) largest = largestLabel(data_DHW, largest);
      await pause(() => labels.drawn());
    }
    labels.drawn();
    this.unkept.clear();
    for (const [path, { bytes }] of chunks) this.packed.set(path, bytes);
    let replayed = 0;
    let error = null;
    for (const { sequence, entry } of entries) {
      const edit = edits.get(entry.edit);
      if (edit === undefined) {
        error = new Error(
          `IndexedDB's journal entry ${sequence} is of edit ${entry.edit}, which it does not hold, so the labels are ` +
            "brought back as they were before it.",
        );
        break;
      }
      for (const [index, runs] of piecesOf(edit)) {
        if (!labels.shown) return null;
        labels.putRuns(index, entry.back ? runs : setTo(runs, edit.value), true);
        for (const label of entry.back ? runs.values_N : [edit.value]) if (label > largest) largest = label;
        await pause(() => labels.drawn());
      }
      replayed += 1;
    }
    labels.drawn();
    if (entries.length > 0) this.sequence = entries.at(-1).sequence + 1;
    for (const id of [...edits.keys(), ...entries.map(({ entry }) => entry.edit), ...head.done, ...head.undone])
      if (id >= this.next) this.next = id + 1;
    const unchecked = new Map();
    for (const [id, edit] of edits) unchecked.set(id, { id, ...edit, checked: false });
    labels.history.restore(unchecked, head);
    return { largest, replayed, error };
  }

  async keep(pause) {
    const labels = this.owner;
    if (labels.dropped || !labels.idle()) return;
    labels.drawn();
    if (this.unkept.size === 0 && labels.history.forgotten.length === 0) return;
    const last = this.sequence - 1;
    const forgotten = labels.history.dropped();
    const paths = this.unkept;
    this.unkept = new Set();
    const chunks = new Map();
    try {
      for (const path of paths) {
        while (labels.edit !== null) await labels.ended.promise;
        if (labels.dropped) return;
        const { level, chunk_ZYX } = pathChunk(path);
        const data_DHW = labels.volume.chunk(level, chunk_ZYX);
        const bytes = empty(data_DHW) ? null : await through(data_DHW, new CompressionStream("gzip"));
        chunks.set(path, bytes);
        // A chunk written since the compaction began may have been written since it was copied.
        if (!this.unkept.has(path)) this.packed.set(path, bytes);
        await pause();
      }
      if (!labels.dropped) await this.keeping.compact(labels.url, chunks, last, forgotten);
    } catch (error) {
      for (const path of paths) this.unkept.add(path);
      labels.history.refused(forgotten);
      throw error;
    }
  }
}
