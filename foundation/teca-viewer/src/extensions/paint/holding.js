import { readLabels } from "./file.js";
import { Labels } from "./labels.js";

export class Holding {
  constructor(viewer, keeping, brush, zeroed, panel) {
    this.viewer = viewer;
    this.keeping = keeping;
    this.brush = brush;
    this.zeroed = zeroed;
    this.panel = panel;
    this.ct = null;
    this.labels = null;
    this.busy = false;
    this.restoring = false;
    this.unrestored = null;
    this.restored = Promise.resolve();
    this.working = Promise.resolve();
    this.waiting = null;
    this.packing = Promise.resolve();
  }

  make() {
    this.labels ??= new Labels(this.viewer, this.ct, this.keeping, this.zeroed);
  }

  keep(mine) {
    this.packing = this.packing
      .then(() => this.brush.settled())
      .then(() => mine.kept.keep(this.brush.pause))
      .catch(this.panel.failed);
  }

  clear() {
    const mine = this.labels;
    mine.drop();
    this.labels = null;
    this.keeping.replace(mine.url, new Map()).catch(this.panel.failed);
  }

  take() {
    this.waiting.take();
    this.waiting.note = "Asked the other tab for this CT's labels; they come here once it has written its edits.";
    this.panel.describe();
  }

  async transfer(work) {
    const { viewer, panel } = this;
    const opened = this.ct;
    const resume = panel.brushSet();
    let message = "";
    const { promise, resolve } = Promise.withResolvers();
    this.working = promise;
    this.busy = true;
    if (resume) viewer.setTool(null);
    panel.describe();
    try {
      await this.brush.settled();
      if (this.ct === opened && !this.restoring) await work();
    } catch (error) {
      panel.failed(error);
      message = error.message;
    } finally {
      this.busy = false;
      if (resume && this.ct === opened && !this.restoring && this.labels !== null && !panel.brushSet())
        panel.setBrush();
      panel.describe(message);
      resolve();
    }
  }

  load(chosen) {
    void this.transfer(async () => {
      const opened = this.ct;
      const made = this.labels === null;
      this.make();
      const mine = this.labels;
      try {
        const candidate = await readLabels(new Uint8Array(await chosen.arrayBuffer()), mine.levels, mine.units_ZYX);
        await this.packing;
        if (this.ct !== opened) {
          if (made) mine.drop();
          return;
        }
        await this.keeping.replace(mine.url, candidate.packed);
        await mine.replace(candidate, this.brush.pause);
        this.panel.raise(candidate.classes);
        this.panel.listClasses();
      } catch (error) {
        if (made && this.labels === mine) {
          mine.drop();
          this.labels = null;
        }
        throw error;
      }
    });
  }

  async restore(opened, wait) {
    const { viewer, keeping, brush, zeroed, panel } = this;
    let message = "";
    let mine = null;
    try {
      await keeping.hold(
        opened.rootUrl,
        (take) => {
          if (this.ct !== opened) return;
          this.waiting = { take, note: wait };
          panel.describe();
        },
        () => void this.giveUp(opened.rootUrl),
      );
      if (this.ct !== opened) return;
      this.waiting = null;
      const { chunks, entries, edits, head } = await keeping.read(opened.rootUrl);
      if ((chunks.size === 0 && entries.length === 0 && edits.size === 0) || this.ct !== opened) return;
      mine = new Labels(viewer, opened, keeping, zeroed);
      this.labels = mine;
      const restored = await mine.kept.restore(chunks, entries, edits, head, brush.pause);
      if (this.ct !== opened) {
        mine.drop();
        return;
      }
      const { largest, replayed, error } = restored;
      panel.raise(largest);
      brush.follow(mine);
      this.keep(mine);
      panel.listClasses();
      if (error !== null) panel.failed(error);
      const brought = [];
      if (chunks.size > 0) brought.push(`${chunks.size} chunk${chunks.size === 1 ? "" : "s"}`);
      if (replayed > 0) {
        const after = chunks.size === 0 ? "" : ` made after ${chunks.size === 1 ? "it" : "them"}`;
        brought.push(`${replayed} edit${replayed === 1 ? "" : "s"}${after}`);
      }
      const { done, undone } = mine.history;
      const undo = done.length === 0 ? "" : `, ${done.length} to undo`;
      const redo = undone.length === 0 ? "" : `, ${undone.length} to redo`;
      const what = brought.length === 0 ? "the edits" : brought.join(" and ");
      message = `Brought back ${what} painted here before${undo}${redo}. Clear all forgets them.`;
    } catch (error) {
      mine?.drop();
      if (this.ct === opened) {
        this.labels = null;
        this.unrestored =
          `The labels this browser keeps for this CT could not be brought back: ${error.message} ` +
          "Reload to try again.";
        keeping.letGo(opened.rootUrl);
      }
      throw error;
    } finally {
      if (this.ct === opened) {
        if (this.unrestored === null) this.restoring = false;
        panel.describe(message);
      }
    }
  }

  async giveUp(url) {
    await this.restored;
    const opened = this.ct;
    if (opened?.rootUrl !== url || this.restoring) return;
    this.restoring = true;
    if (this.panel.brushSet()) this.viewer.setTool(null);
    const mine = this.labels;
    this.labels = null;
    this.panel.describe();
    await this.brush.settled();
    await this.working;
    if (mine !== null && this.keeping.lost.has(url)) {
      this.keep(mine);
      await this.packing;
      if (this.ct === opened && this.keeping.lost.has(url)) {
        this.labels = mine;
        this.restoring = false;
        this.panel.describe(
          "Another tab asked for this CT's labels, but IndexedDB has not written an edit of them, so they stay here; " +
            "the status bar says why.",
        );
        return;
      }
    }
    mine?.drop();
    await this.packing;
    if (this.ct !== opened) return;
    this.keeping.letGo(url);
    this.restored = this.restore(opened, "Another tab took this CT's labels; Paint here takes them back.").catch(
      this.panel.failed,
    );
  }

  open(opened) {
    if (this.panel.brushSet()) this.viewer.setTool(null);
    this.labels?.hide();
    const closed = this.ct;
    this.ct = opened;
    this.labels = null;
    this.restoring = true;
    this.unrestored = null;
    this.waiting = null;
    this.panel.describe();
    this.restored = this.working
      .then(() => this.packing)
      .then(() => {
        if (closed !== null && closed.rootUrl !== opened.rootUrl) this.keeping.letGo(closed.rootUrl);
        return this.restore(
          opened,
          "Another tab has this CT's labels open; Paint here takes them, or they come here once it closes them.",
        );
      })
      .catch(this.panel.failed);
  }
}
