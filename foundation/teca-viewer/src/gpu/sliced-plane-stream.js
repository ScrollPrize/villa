import { CT_BRICKS_PER_FRAME, PREFETCH_CHUNK_PRIORITY, SHOWN_CHUNK_PRIORITY_LIMIT } from "../config.js";
import { BrickReads } from "./brick-reads.js";
import { rebuildPageTable } from "./pane-slice.js";
import { planeReads, retainPlaneReads } from "./plane-reads.js";

const HIDDEN_PLANE_UPLOAD_PRIORITY = 2;

export class SlicedPlaneStream {
  constructor({ panes, uploadBudget }) {
    this.panes = panes;
    this.uploadBudget = uploadBudget;
    this.reads = null;
    this.onError = null;
    this.admissionTasks = [];
    this.admissionKeys = new Set();
    this.uploadTasks = [];
    this.uploadKeys = new Map();
    this.uploadPriority = Infinity;
  }

  read(volume, channel, plans, onError) {
    retainPlaneReads(volume, channel, plans);
    volume.cancelPendingChunks(0, SHOWN_CHUNK_PRIORITY_LIMIT - 1, true);
    volume.cancelPendingChunks(PREFETCH_CHUNK_PRIORITY, Infinity, true);
    this.onError = onError;
    if (this.reads?.volume !== volume) {
      this.reads?.dispose();
      this.reads = new BrickReads({
        volume,
        channel,
        wants: (task) => this.acceptsAdmission(task),
        deliver: (task, chunk_DHW) =>
          new Promise((resolve) => this.enqueueAdmission({ ...task, chunk_DHW, onComplete: resolve })),
        onError: (error) => this.onError(error),
      });
    }
    this.reads.want(planeReads(plans));
  }

  retain(plans) {
    if (this.reads === null) return false;
    retainPlaneReads(this.reads.volume, this.reads.channel, plans);
    return true;
  }

  refresh(residentKey, data_DHW) {
    for (const task of this.admissionTasks) if (task.residentKey === residentKey) task.chunk_DHW = data_DHW;
  }

  clear() {
    this.reads?.dispose();
    this.reads = null;
    for (const task of this.admissionTasks) task.onComplete();
    this.admissionTasks = [];
    this.admissionKeys.clear();
    this.uploadTasks = [];
    this.uploadKeys.clear();
  }

  stats() {
    return {
      pendingFetches: this.reads?.pending() ?? 0,
      uploadQueue: this.admissionTasks.length + this.uploadTasks.length,
    };
  }

  upload(pane, level, chunk_ZYX, entry) {
    if (
      !pane.visible() ||
      !level.atlas.inWorkingSet(entry.key) ||
      (pane.hasAdmission(level.scaleKey, chunk_ZYX, entry) && !level.atlas.needsUpload(pane.renderer, entry))
    )
      return false;
    if (entry.slot === null) return pane.admit(level.scaleKey, chunk_ZYX, entry);
    const uploadKey = `${level.atlas.texture.id}\0${entry.generation}`;
    const queued = this.uploadKeys.get(uploadKey);
    if (queued !== undefined) queued.shown ||= pane.shown;
    else {
      const task = {
        uploadKey,
        residentKey: entry.key,
        renderer: pane.renderer,
        level,
        chunk_ZYX,
        entry,
        shown: pane.shown,
      };
      this.uploadKeys.set(uploadKey, task);
      this.uploadTasks.push(task);
    }
    this.scheduleUploads(pane.shown);
    return false;
  }

  enqueueAdmission(task) {
    if (this.admissionKeys.has(task.key) || !this.acceptsAdmission(task)) {
      task.onComplete();
      return;
    }
    this.admissionKeys.add(task.key);
    const hidden = task.shown ? this.admissionTasks.findIndex((queued) => !queued.shown) : -1;
    if (hidden < 0) this.admissionTasks.push(task);
    else this.admissionTasks.splice(hidden, 0, task);
    this.scheduleUploads(task.shown);
  }

  acceptsAdmission(task) {
    for (const pane of this.panes()) {
      const level = pane.level(task.scaleKey);
      if (pane.visible() && level?.atlas.inWorkingSet(task.residentKey) && level.atlas.get(task.residentKey) === null)
        return true;
    }
    return false;
  }

  prune() {
    this.reads?.prune();
    this.admissionTasks = this.admissionTasks.filter((task) => {
      if (this.acceptsAdmission(task)) return true;
      task.onComplete();
      return false;
    });
    this.admissionKeys = new Set(this.admissionTasks.map((task) => task.key));
    this.uploadTasks = this.uploadTasks.filter(
      (task) =>
        task.level.atlas.isResident(task.entry) &&
        task.level.atlas.inWorkingSet(task.residentKey) &&
        [...this.panes()].some((pane) => pane.accepts(task.level.scaleKey, task.chunk_ZYX)),
    );
    this.uploadKeys = new Map(this.uploadTasks.map((task) => [task.uploadKey, task]));
  }

  scheduleUploads(shown) {
    const priority = shown ? 0 : HIDDEN_PLANE_UPLOAD_PRIORITY;
    if (priority >= this.uploadPriority) return;
    this.uploadPriority = priority;
    this.uploadBudget.schedule(this, priority, (timestamp) => this.processUploads(timestamp));
  }

  processUploads(timestamp) {
    this.uploadPriority = Infinity;
    const uploadBudget = this.uploadBudget;
    const presented = new Set();
    const dirtyLevels = new Set();
    let processedAdmissions = 0;
    const admittedAtlases = new Set();
    const admit = (task) => {
      this.admissionKeys.delete(task.key);
      const acceptingPanes = [...this.panes()].filter((pane) => {
        const level = pane.level(task.scaleKey);
        return pane.visible() && level?.atlas.inWorkingSet(task.residentKey);
      });
      for (const pane of acceptingPanes) {
        const level = pane.level(task.scaleKey);
        const residentEntry = level.atlas.get(task.residentKey);
        const entry = residentEntry ?? level.atlas.admit(task.residentKey, task.chunk_DHW);
        if (entry === null) continue;
        if (residentEntry === null) admittedAtlases.add(level.atlas);
        if (this.upload(pane, level, task.chunk_ZYX, entry)) {
          dirtyLevels.add(level);
          presented.add(pane);
        }
      }
      task.onComplete();
    };
    for (const task of this.admissionTasks) if (task.chunk_DHW === null) admit(task);
    this.admissionTasks = this.admissionTasks.filter((task) => task.chunk_DHW !== null);
    while (
      processedAdmissions < CT_BRICKS_PER_FRAME &&
      this.admissionTasks.length > 0 &&
      uploadBudget.available(timestamp) > 0
    ) {
      processedAdmissions += 1;
      admit(this.admissionTasks.shift());
    }
    for (const pane of this.panes()) if (pane.dropEvicted(admittedAtlases)) presented.add(pane);
    let processedUploads = 0;
    const uploadRenderers = new Set();
    while (processedUploads < CT_BRICKS_PER_FRAME && this.uploadTasks.length > 0) {
      const task = this.uploadTasks.shift();
      if (
        !task.level.atlas.isResident(task.entry) ||
        !task.level.atlas.inWorkingSet(task.residentKey) ||
        ![...this.panes()].some((pane) => pane.accepts(task.level.scaleKey, task.chunk_ZYX))
      ) {
        this.uploadKeys.delete(task.uploadKey);
        continue;
      }
      const remainingBytes = uploadBudget.available(timestamp);
      if (remainingBytes < task.level.atlas.formatDescriptor.bytesPerVoxel) {
        this.uploadTasks.unshift(task);
        break;
      }
      if (!uploadRenderers.has(task.renderer)) {
        task.renderer.resetState();
        uploadRenderers.add(task.renderer);
      }
      const upload = task.level.atlas.uploadPart(task.renderer, task.entry, remainingBytes);
      processedUploads += 1;
      uploadBudget.spend(upload.uploadedBytes);
      if (!upload.complete) {
        this.uploadTasks.push(task);
        continue;
      }
      this.uploadKeys.delete(task.uploadKey);
      for (const pane of this.panes()) {
        const level = pane.level(task.level.scaleKey);
        if (level?.atlas !== task.level.atlas) continue;
        if (pane.admit(task.level.scaleKey, task.chunk_ZYX, task.entry)) dirtyLevels.add(level);
        else if (!pane.hasAdmission(task.level.scaleKey, task.chunk_ZYX, task.entry)) continue;
        presented.add(pane);
      }
    }
    for (const level of dirtyLevels) rebuildPageTable(level);
    for (const pane of presented) pane.present();
    if (this.admissionTasks.length > 0 || this.uploadTasks.length > 0)
      this.scheduleUploads(
        this.admissionTasks.some((task) => task.shown) || this.uploadTasks.some((task) => task.shown),
      );
  }
}
