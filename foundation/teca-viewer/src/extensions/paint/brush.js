import { chordSteps } from "../../math.js";
import { AXIS_TOLERANCE, planeFill } from "./fill.js";
import { voxelTiles } from "./stamp.js";
import { meshSwept, meshTriangles } from "./surface-sweep.js";
import { sweptTiles, tilesOf } from "./sweep.js";
import { slices } from "./tasks.js";

// Of the smallest radius across the pane, how far the pointer moves before the brush sweeps to it: a curve of radius R
// is cut to chords this long, whose outside falls short of it by (spacing)² / 8R, 0.8% of the radius where R is the
// radius.
const SPACING = 1 / 4;
// A point this near the stroke's last plane, in finest voxels, is on it: a slice pane's points are off it by rounding.
const ON_PLANE = 1e-6;
const FILL_LIMIT = 2 ** 20;
const SURFACE_SAG = 1;
const CELLS = 2 ** 16;

function voxelAt(point_ZYX, { downsample_ZYX, offset_ZYX }) {
  return point_ZYX.map(
    (world, axis) => (world - offset_ZYX[axis] - (downsample_ZYX[axis] - 1) / 2) / downsample_ZYX[axis],
  );
}

export class Brush {
  constructor(radii_ZYX, failed, said, surfacePoints, sweepers) {
    this.radii_ZYX = radii_ZYX;
    this.failed = failed;
    this.said = said;
    this.surfacePoints = surfacePoints;
    this.sweepers = sweepers;
    this.sweeps = 0;
    this.finding = Promise.resolve();
    this.painting = Promise.resolve();
    this.coarsening = Promise.resolve();
    this.stroke = null;
    this.pause = slices();
  }

  resize(radii_ZYX) {
    this.radii_ZYX = radii_ZYX;
  }

  queue(work) {
    this.painting = this.painting.then(work).catch(this.failed);
  }

  follow(labels) {
    this.coarsening = this.coarsening.then(() => labels.followed(this.pause)).catch(this.failed);
  }

  settled() {
    return this.painting.then(() => this.coarsening);
  }

  begin(labels, gate, value) {
    const ended = Promise.withResolvers();
    this.stroke = { labels, gate, value, last: null, surfaceAt: null, vertices: new Map(), ended };
    this.queue(() => labels.beginStroke());
    return ended.promise;
  }

  async stamped(labels, tiles, value, gate, pause) {
    for (const tile of tiles) {
      labels.stamp(tile, [0, 0, 0], value, gate);
      await pause(() => labels.drawn());
    }
    labels.drawn();
    this.follow(labels);
  }

  to(at, released) {
    const stroke = this.stroke;
    const { pane, cell_HW } = at;
    if (pane === "surface") stroke.surfaceAt ??= this.surfacePoints();
    const { point_ZYX, normal_ZYX } =
      pane === "surface" ? (stroke.surfaceAt(cell_HW) ?? { point_ZYX: null, normal_ZYX: null }) : at;
    if (point_ZYX === null) {
      stroke.last = null;
      return;
    }
    const { labels, gate, value, last } = stroke;
    const at_ZYX = voxelAt(point_ZYX, labels.volume.scales[0]);
    const radii_ZYX = this.radii_ZYX;
    if (last !== null) {
      const length = Math.hypot(...at_ZYX.map((value, axis) => value - last.at_ZYX[axis]));
      const across = radii_ZYX.filter((_, axis) => Math.abs(normal_ZYX[axis]) < 1 - AXIS_TOLERANCE);
      if (released ? length === 0 : length < Math.max(1, Math.min(...across) * SPACING)) return;
    }
    // A release's segment, a quarter of the radius at most, is made in the task it begins in, 0.4 to 2.2 ms at radius
    // 256 on a plane and 5 to 7 on the Surface, where a pause would wait for a frame, so the stroke's journal entry is
    // asked for then.
    const pause = released ? async () => {} : this.pause;
    if (pane === "surface") {
      this.along(cell_HW, at_ZYX, released, pause);
      return;
    }
    const level = normal_ZYX.reduce((total, component, axis) => total + component * at_ZYX[axis], 0);
    const kept =
      last?.plane.normal_ZYX.every((component, axis) => component === normal_ZYX[axis]) &&
      Math.abs(last.plane.level - level) < ON_PLANE;
    const plane = kept ? last.plane : { normal_ZYX, level };
    const to_ZYX = kept ? at_ZYX.map((value, axis) => value - (level - plane.level) * normal_ZYX[axis]) : at_ZYX;
    const before = last !== null && last.radii_ZYX === radii_ZYX ? last.plane : null;
    const from_ZYX = last === null ? to_ZYX : last.at_ZYX;
    stroke.last = { at_ZYX: to_ZYX, cell_HW: null, plane, radii_ZYX };
    this.queue(async () => {
      const { shape_ZYX } = labels.volume.scales[0];
      const swept = await sweptTiles(radii_ZYX, plane, from_ZYX, to_ZYX, before, shape_ZYX, pause);
      if (swept.tiles.length === 0) return;
      if (gate !== null) await gate.load(swept.low_ZYX, swept.high_ZYX);
      await this.stamped(labels, swept.tiles, value, gate, pause);
    });
  }

  along(cell_HW, at_ZYX, released, pause) {
    const stroke = this.stroke;
    const { labels, gate, value, last, surfaceAt, vertices } = stroke;
    const radii_ZYX = this.radii_ZYX;
    const pointAt = (cell_HW) => {
      const point_ZYX = surfaceAt(cell_HW)?.point_ZYX ?? null;
      return point_ZYX === null ? null : voxelAt(point_ZYX, labels.volume.scales[0]);
    };
    const vertexAt = (h, w) => {
      const key = h * CELLS + w;
      if (!vertices.has(key)) vertices.set(key, pointAt([h, w]));
      return vertices.get(key);
    };
    const from = last ?? { at_ZYX, cell_HW };
    const cellAt = (t) => from.cell_HW.map((value, axis) => value + (cell_HW[axis] - value) * t);
    const cells = Math.hypot(...cell_HW.map((value, axis) => value - from.cell_HW[axis]));
    const steps = chordSteps((t) => pointAt(cellAt(t)), from.at_ZYX, at_ZYX, SURFACE_SAG, cells);
    stroke.last = { at_ZYX, cell_HW, plane: null, radii_ZYX };
    for (const [index, { t, point_ZYX: to_ZYX }] of steps.entries()) {
      const from_ZYX = index === 0 ? from.at_ZYX : steps[index - 1].point_ZYX;
      const before = index > 0 || (last !== null && last.radii_ZYX === radii_ZYX);
      const step_HW = cellAt(t);
      const found = this.finding.then(() =>
        meshTriangles(radii_ZYX, vertexAt, step_HW, from_ZYX, to_ZYX, before, pause),
      );
      this.finding = found.catch(() => {});
      const packed = found.then((points_N9) => {
        const job = { radii_ZYX, points_N9, from_ZYX, to_ZYX, before, shape_ZYX: labels.volume.scales[0].shape_ZYX };
        if (released) return meshSwept(job);
        const sweeper = this.sweepers[this.sweeps % this.sweepers.length];
        this.sweeps += 1;
        return sweeper(job);
      });
      packed.catch(() => {});
      this.queue(async () => {
        const swept = tilesOf(await packed);
        if (swept.tiles.length === 0) return;
        if (gate !== null) await gate.load(swept.low_ZYX, swept.high_ZYX);
        await this.stamped(labels, swept.tiles, value, gate, pause);
      });
    }
  }

  end() {
    const { labels, gate, value, ended } = this.stroke;
    this.stroke = null;
    this.queue(() => {
      labels.endStroke(value);
      gate?.release();
    });
    ended.resolve(this.painting);
    return this.painting;
  }

  fill(labels, gate, { point_ZYX, normal_ZYX }, value) {
    const { shape_ZYX } = labels.volume.scales[0];
    const start_ZYX = voxelAt(point_ZYX, labels.volume.scales[0]).map(Math.round);
    if (start_ZYX.some((voxel, axis) => voxel < 0 || voxel >= shape_ZYX[axis])) return this.painting;
    this.queue(async () => {
      const label = labels.lookup();
      const target = label(...start_ZYX);
      if (target === value) return;
      const open = (z, y, x) => label(z, y, x) === target && (gate === null || gate.passes(z, y, x));
      const load = (voxels_N3) => {
        const low_ZYX = [Infinity, Infinity, Infinity];
        const high_ZYX = [-Infinity, -Infinity, -Infinity];
        for (let index = 0; index < voxels_N3.length; index++) {
          low_ZYX[index % 3] = Math.min(low_ZYX[index % 3], voxels_N3[index]);
          high_ZYX[index % 3] = Math.max(high_ZYX[index % 3], voxels_N3[index] + 1);
        }
        return gate.load(low_ZYX, high_ZYX);
      };
      const voxels_N3 = await planeFill(start_ZYX, normal_ZYX, shape_ZYX, open, load, FILL_LIMIT, this.pause).finally(
        () => gate?.release(),
      );
      if (voxels_N3 === null) {
        this.said(`Paint: a fill takes at most ${FILL_LIMIT} voxels, so this one took none. Close the region first.`);
        return;
      }
      if (voxels_N3.length === 0) {
        this.said("Paint: the CT under the fill's voxel is below the gate, so it took none.");
        return;
      }
      const tiles = await voxelTiles(voxels_N3, this.pause);
      labels.beginStroke();
      // An edit that is begun ends, or a compaction would wait for it.
      try {
        await this.stamped(labels, tiles, value, null, this.pause);
      } finally {
        labels.endStroke(value);
      }
    });
    return this.painting;
  }
}
