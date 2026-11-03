import { chunkPath, pathChunk, storeDocuments } from "./file.js";
import { History, Pieces, piecesOf, Runs, setTo } from "./history.js";
import { Kept } from "./kept.js";
import { Pyramid } from "./pyramid.js";
import { cropped, MAX_SIDE } from "./stamp.js";

const NAME = "paint";
const LABEL_OPACITY = 0.6;
// The steps of undo and redo held: a stroke of radius 256 over other labels holds 0.23 MiB of runs on an xy or a zy
// plane and 1.8 MiB on the plane across (1, 1, 1), whose runs are mostly a voxel long.
const HISTORY_BYTES = 128 * 2 ** 20;

export class Labels {
  constructor(viewer, ct, keeping, zeroed) {
    this.volume = viewer.memoryVolume(NAME, zeroed);
    const { shape_ZYX } = this.volume.scales[0];
    if (shape_ZYX.some((size) => size > MAX_SIDE))
      throw new Error(`CT level 0 is ${shape_ZYX.join("×")} voxels; paint takes at most ${MAX_SIDE} a side.`);
    this.pyramid = new Pyramid(this.volume);
    this.handle = viewer.addLayer(
      { slices: this.volume, surface: this.volume },
      { color_RGB: null, opacity: LABEL_OPACITY },
    );
    this.url = ct.rootUrl;
    this.levels = this.volume.scales.map(({ shape_ZYX, chunks_ZYX }, level) => {
      const { scale_ZYX, translation_ZYX } = ct.scales[level];
      return { shape_ZYX, chunks_ZYX, scale_ZYX, translation_ZYX };
    });
    this.units_ZYX = ct.metadata.physicalUnits_ZYX;
    this.held = new Set();
    this.unsaved = new Set();
    this.grid_ZYX = this.volume.scales[0].shape_ZYX.map((size, axis) =>
      Math.ceil(size / this.volume.scales[0].chunks_ZYX[axis]),
    );
    const [, chunkY, chunkX] = this.volume.scales[0].chunks_ZYX;
    this.strides = [chunkY * chunkX, chunkX, 1];
    this.edit = null;
    this.ended = null;
    this.history = new History(HISTORY_BYTES);
    this.kept = new Kept(this, keeping);
    this.writing = new Runs(2, 1);
    this.shown = true;
    this.dropped = false;
  }

  hide() {
    this.shown = false;
  }

  drop() {
    if (this.shown) this.handle.remove();
    this.shown = false;
    this.dropped = true;
    this.pyramid.forget();
  }

  beginStroke() {
    this.edit = new Pieces();
    this.ended = Promise.withResolvers();
  }

  endStroke(value) {
    const pieces = this.ending();
    if (pieces.count === 0) return;
    const id = this.kept.newId();
    this.history.record({ id, value, ...pieces.packed(), checked: true });
    this.kept.journal({ id, value, ...pieces.views() }, false, true);
  }

  idle() {
    return this.edit === null && this.pyramid.pending.size === 0;
  }

  ending() {
    const edit = this.edit;
    this.edit = null;
    this.ended.resolve();
    return edit;
  }

  chunkAt(index) {
    const [, gridY, gridX] = this.grid_ZYX;
    return [Math.floor(index / (gridY * gridX)), Math.floor(index / gridX) % gridY, index % gridX];
  }

  lookup() {
    const { chunks_ZYX } = this.volume.scales[0];
    const [, gridY, gridX] = this.grid_ZYX;
    let index = -1;
    let data_DHW = null;
    return (z, y, x) => {
      const [cz, cy, cx] = [
        Math.floor(z / chunks_ZYX[0]),
        Math.floor(y / chunks_ZYX[1]),
        Math.floor(x / chunks_ZYX[2]),
      ];
      if ((cz * gridY + cy) * gridX + cx !== index) {
        index = (cz * gridY + cy) * gridX + cx;
        data_DHW = this.volume.chunk(0, [cz, cy, cx]);
      }
      if (data_DHW === null) return 0;
      return data_DHW[
        ((z - cz * chunks_ZYX[0]) * chunks_ZYX[1] + y - cy * chunks_ZYX[1]) * chunks_ZYX[2] + x - cx * chunks_ZYX[2]
      ];
    };
  }

  stamp(tile, centre_ZYX, value, gate) {
    const { shape_ZYX: level_ZYX, chunks_ZYX } = this.volume.scales[0];
    const start_ZYX = tile.from_ZYX.map((from, axis) => centre_ZYX[axis] + from);
    const low_ZYX = start_ZYX.map((start) => Math.max(0, start));
    const high_ZYX = start_ZYX.map((start, axis) => Math.min(level_ZYX[axis], start + tile.shape_ZYX[axis]));
    if (low_ZYX.some((low, axis) => low >= high_ZYX[axis])) return;
    const { axis, runs_N4 } = tile;
    const a = axis === 0 ? 1 : 0;
    const b = axis === 2 ? 1 : 2;
    const startA = start_ZYX[a];
    const startB = start_ZYX[b];
    const startR = start_ZYX[axis];
    const lowA = low_ZYX[a];
    const lowB = low_ZYX[b];
    const lowR = low_ZYX[axis];
    const highA = high_ZYX[a];
    const highB = high_ZYX[b];
    const highR = high_ZYX[axis];
    const sizeA = chunks_ZYX[a];
    const sizeB = chunks_ZYX[b];
    const sizeR = chunks_ZYX[axis];
    const [, gridY, gridX] = this.grid_ZYX;
    const grid_ZYX = [gridY * gridX, gridX, 1];
    const gridA = grid_ZYX[a];
    const gridB = grid_ZYX[b];
    const gridR = grid_ZYX[axis];
    const strideA = this.strides[a];
    const strideB = this.strides[b];
    const stride = this.strides[axis];
    this.writing.reset(axis, stride);
    const first_ZYX = low_ZYX.map((low, axis) => Math.floor(low / chunks_ZYX[axis]));
    const one = first_ZYX.every((chunk, axis) => (chunk + 1) * chunks_ZYX[axis] >= high_ZYX[axis]);
    let key = -1;
    let chunk_ZYX = null;
    let data_DHW = null;
    let entered = false;
    let lowered = false;
    const voxel_ZYX = [0, 0, 0];
    for (let run = 0; run < runs_N4.length; run += 4) {
      const across = startA + runs_N4[run + a];
      const beside = startB + runs_N4[run + b];
      if (across < lowA || across >= highA || beside < lowB || beside >= highB) continue;
      const chunkA = one ? first_ZYX[a] : Math.floor(across / sizeA);
      const chunkB = one ? first_ZYX[b] : Math.floor(beside / sizeB);
      const base = (across - chunkA * sizeA) * strideA + (beside - chunkB * sizeB) * strideB;
      const end = Math.min(highR, startR + runs_N4[run + axis] + runs_N4[run + 3]);
      for (let from = Math.max(lowR, startR + runs_N4[run + axis]); from < end; ) {
        const chunkR = one ? first_ZYX[axis] : Math.floor(from / sizeR);
        if (key !== chunkA * gridA + chunkB * gridB + chunkR * gridR) {
          if (key >= 0) this.setRuns(chunk_ZYX, lowered ? null : value);
          lowered = false;
          key = chunkA * gridA + chunkB * gridB + chunkR * gridR;
          chunk_ZYX = [0, 0, 0];
          chunk_ZYX[a] = chunkA;
          chunk_ZYX[b] = chunkB;
          chunk_ZYX[axis] = chunkR;
          data_DHW = this.volume.chunk(0, chunk_ZYX);
          entered = false;
        }
        const stop = one ? end : Math.min(end, (chunkR + 1) * sizeR);
        let first = 0;
        let length = 0;
        let held = 0;
        for (let along = from, at = base + (from - chunkR * sizeR) * stride; along < stop; along++, at += stride) {
          const was = data_DHW === null ? 0 : data_DHW[at];
          if (was === value) continue;
          if (gate !== null) {
            voxel_ZYX[a] = across;
            voxel_ZYX[b] = beside;
            voxel_ZYX[axis] = along;
            if (!gate.passes(voxel_ZYX[0], voxel_ZYX[1], voxel_ZYX[2])) continue;
          }
          if (length > 0 && was === held && at === first + length * stride) {
            length += 1;
            continue;
          }
          if (length > 0) {
            this.edit.add(first, length, held);
            this.writing.add(first, length, value);
          } else if (!entered) {
            this.edit.enter(key, axis, stride);
            entered = true;
          }
          if (was > value) lowered = true;
          first = at;
          length = 1;
          held = was;
        }
        if (length > 0) {
          this.edit.add(first, length, held);
          this.writing.add(first, length, value);
        }
        from = stop;
      }
    }
    this.setRuns(chunk_ZYX, lowered ? null : value);
  }

  setRuns(chunk_ZYX, raised) {
    const { axis, offsets_N, lengths_N, values_N, count } = this.writing;
    if (count === 0) return;
    const [offsets, lengths] = [offsets_N.subarray(0, count), lengths_N.subarray(0, count)];
    this.volume.writeRuns(0, chunk_ZYX, axis, offsets, lengths, values_N.subarray(0, count));
    this.pyramid.followRuns(chunk_ZYX, axis, offsets, lengths, raised, false);
    this.writing.clear();
  }

  put(path, data_DHW) {
    const { level, chunk_ZYX } = pathChunk(path);
    const { shape_ZYX, chunks_ZYX } = this.volume.scales[level];
    const start_ZYX = chunk_ZYX.map((chunk, axis) => chunk * chunks_ZYX[axis]);
    const size_ZYX = start_ZYX.map((start, axis) => Math.min(chunks_ZYX[axis], shape_ZYX[axis] - start));
    if (size_ZYX.every((size, axis) => size === chunks_ZYX[axis])) this.volume.place(level, chunk_ZYX, data_DHW);
    else this.volume.write(level, start_ZYX, size_ZYX, cropped(data_DHW, chunks_ZYX, [0, 0, 0], size_ZYX));
  }

  drawn() {
    const changes = this.volume.take();
    for (const { scale, chunk_ZYX } of changes) {
      const path = chunkPath(this.volume.scales.indexOf(scale), chunk_ZYX);
      this.held.add(path);
      this.unsaved.add(path);
      this.kept.written(path);
    }
    if (this.shown) this.handle.changed(changes);
  }

  followed(pause) {
    return this.pyramid.followed(pause, () => this.drawn());
  }

  async turn(back, pause) {
    const edit = back ? this.history.undo() : this.history.redo();
    if (edit === null) return false;
    if (!edit.checked) {
      if (!this.holds(edit, back)) {
        this.history.forget();
        this.kept.forget();
        throw new Error(
          `the labels are not as edit ${edit.id} kept in this browser ${back ? "left" : "found"} them, so undo's ` +
            "history is forgotten.",
        );
      }
      edit.checked = true;
    }
    this.kept.journal(edit, back, false);
    this.edit = new Pieces();
    this.ended = Promise.withResolvers();
    try {
      for (const [index, runs] of piecesOf(edit)) {
        this.putRuns(index, back ? runs : setTo(runs, edit.value), false);
        await pause(() => this.drawn());
      }
    } finally {
      this.ending();
    }
    this.drawn();
    return true;
  }

  holds(edit, back) {
    for (const [index, { axis, offsets_N, lengths_N, values_N }] of piecesOf(edit)) {
      const data_DHW = this.volume.chunk(0, this.chunkAt(index));
      const stride = this.strides[axis];
      for (let run = 0; run < offsets_N.length; run++) {
        const label = back ? edit.value : values_N[run];
        for (let left = lengths_N[run], voxel = offsets_N[run]; left > 0; left--, voxel += stride)
          if ((data_DHW === null ? 0 : data_DHW[voxel]) !== label) return false;
      }
    }
    return true;
  }

  putRuns(index, { axis, offsets_N, lengths_N, values_N }, mixed) {
    const chunk_ZYX = this.chunkAt(index);
    this.volume.writeRuns(0, chunk_ZYX, axis, offsets_N, lengths_N, values_N);
    this.pyramid.followRuns(chunk_ZYX, axis, offsets_N, lengths_N, null, mixed);
  }

  async replace({ chunks, packed }, pause) {
    for (const path of [...this.held].filter((path) => !chunks.has(path))) {
      const { chunks_ZYX } = this.volume.scales[pathChunk(path).level];
      this.put(path, new Uint8Array(chunks_ZYX[0] * chunks_ZYX[1] * chunks_ZYX[2]));
      await pause(() => this.drawn());
    }
    for (const [path, data_DHW] of chunks) {
      this.put(path, data_DHW);
      await pause(() => this.drawn());
    }
    this.drawn();
    this.unsaved.clear();
    this.kept.replaced(packed);
    this.history = new History(HISTORY_BYTES);
  }

  async savedEntries() {
    const paths = this.unsaved;
    this.unsaved = new Set();
    try {
      const encoder = new TextEncoder();
      const entries = Object.entries(storeDocuments(this.levels, this.units_ZYX)).map(([name, value]) => ({
        name,
        bytes: encoder.encode(JSON.stringify(value)),
      }));
      for (const path of this.held) {
        const bytes = await this.kept.packedChunk(path);
        if (bytes !== null) entries.push({ name: path, bytes });
      }
      return entries;
    } catch (error) {
      for (const path of paths) this.unsaved.add(path);
      throw error;
    }
  }
}
