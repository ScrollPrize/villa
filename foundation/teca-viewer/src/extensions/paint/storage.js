import { lock } from "../../storage.js";

export const CHUNKS = "chunks";
export const EDITS = "edits";
export const HEADS = "heads";
export const JOURNAL = "journal";
export const DATABASE = {
  name: "teca-paint",
  version: 1,
  upgrade: (opened) => {
    for (const store of [CHUNKS, EDITS, HEADS, JOURNAL]) opened.createObjectStore(store);
  },
};

// A journal entry IndexedDB has not written by then is taken as failed, so the note says not kept rather than being
// kept for good; of 2,269 written on the real scroll the slowest took 1.15 s, while three workers read the journal.
export const ENTRY_LIMIT_MS = 10_000;

// Workers that read a CT and gunzip a share of its chunks each: a reload brought back the 3,740 of a stroke of radius
// 256 across the Surface zoomed out 26 notches in 2.1 to 2.4 s when the page gunzipped them, and in 1.7 to 1.9 s with
// one worker, 1.2 to 1.4 with two, 1.0 to 1.4 with four and 1.1 to 1.5 with eight.
const READERS = 4;

export const NONE = { done: [], undone: [] };

export function chunkRange(url) {
  return IDBKeyRange.bound([url, ""], [url, "￿"]);
}

export function numberRange(url, last = Infinity) {
  return IDBKeyRange.bound([url, -Infinity], [url, last]);
}

export class Keeping {
  constructor(post, reader, failed, changed) {
    this.post = post;
    this.reader = reader;
    this.failed = failed;
    this.changed = changed;
    this.replies = new Map();
    this.next = 0;
    this.stopped = null;
    this.writes = Promise.resolve();
    this.entries = Promise.resolve();
    this.owed = 0;
    this.lost = new Map();
    this.held = new Map();
  }

  answered({ id, error }) {
    const reply = this.replies.get(id);
    this.replies.delete(id);
    if (error === undefined) reply.resolve();
    else reply.reject(new Error(error));
  }

  stop(error) {
    this.stopped = error;
    for (const reply of this.replies.values()) reply.reject(error);
    this.replies.clear();
  }

  write(url, message, transfer = []) {
    const id = this.next;
    this.next += 1;
    const reply = Promise.withResolvers();
    if (this.stopped !== null) reply.reject(this.stopped);
    else if (this.held.get(url)?.granted !== true)
      reply.reject(new Error(`this page does not hold the labels of ${url}, so it writes none of them.`));
    else {
      this.replies.set(id, reply);
      this.post({ ...message, id }, transfer);
    }
    this.writes = Promise.all([this.writes, reply.promise.catch(() => {})]);
    return reply.promise;
  }

  async hold(url, waiting, asked) {
    if (!this.held.has(url)) this.held.set(url, this.request(url));
    const hold = this.held.get(url);
    hold.asked = asked;
    if (!(await hold.free) && !hold.granted) waiting(hold.take);
    await hold.ready;
  }

  request(url) {
    const name = `teca-paint ${url}`;
    const channel = new BroadcastChannel(name);
    const hold = lock(name);
    const release = hold.release;
    void hold.ready.then(() => {
      channel.onmessage = () => hold.asked();
    });
    Object.assign(hold, {
      take: () => channel.postMessage("take"),
      asked: null,
      release: () => {
        channel.close();
        release();
      },
    });
    return hold;
  }

  letGo(url) {
    const hold = this.held.get(url);
    if (hold === undefined) return;
    this.held.delete(url);
    void this.writes.then(hold.release);
  }

  owe(ended) {
    this.owed += 1;
    this.changed();
    void ended
      .then(() => this.entries)
      .finally(() => {
        this.owed -= 1;
        this.changed();
      });
  }

  journal(url, sequence, entry, head, edit) {
    const { promise: late, reject } = Promise.withResolvers();
    const limit = setTimeout(
      () =>
        reject(
          new Error(
            `IndexedDB has not written journal entry ${sequence} within ${ENTRY_LIMIT_MS / 1000} s, so a reload ` +
              "may lose its edit.",
          ),
        ),
      ENTRY_LIMIT_MS,
    );
    const arrays = edit === null ? [] : [edit.chunks_N3, edit.offsets_N, edit.lengths_N, edit.values_N];
    const asked = this.write(
      url,
      { journal: [url, sequence, entry, head, edit] },
      arrays.map(({ buffer }) => buffer),
    );
    const written = Promise.race([asked, late])
      .catch((error) => {
        this.lost.set(url, sequence);
        this.failed(error);
      })
      .finally(() => clearTimeout(limit));
    this.entries = Promise.all([this.entries, written]);
  }

  forget(url) {
    this.write(url, { forget: url }).catch(this.failed);
  }

  async compact(url, chunks, last, forgotten) {
    await this.write(url, { compact: [url, chunks, last, forgotten] });
    if (this.lost.get(url) <= last) this.lost.delete(url);
  }

  async replace(url, chunks) {
    await this.write(url, { replace: [url, chunks] });
    this.lost.delete(url);
  }

  async read(url) {
    await this.writes;
    const shares = await Promise.all(Array.from({ length: READERS }, (_, part) => this.share(url, part)));
    const chunks = shares.flatMap((share) => share.chunks).sort(([one], [other]) => one - other);
    const { entries, edits, head } = shares[0].kept;
    return {
      chunks: new Map(chunks.map(([, path, data_DHW, bytes]) => [path, { data_DHW, bytes }])),
      entries,
      edits: new Map(edits),
      head,
    };
  }

  share(url, part) {
    const { promise, resolve, reject } = Promise.withResolvers();
    const chunks = [];
    let kept = null;
    const end = this.reader({ read: [url, part, READERS] }, (data) => {
      if (data.chunk !== undefined) chunks.push(data.chunk);
      else if (data.kept !== undefined) kept = data.kept;
      else {
        end();
        if (data.error === undefined) resolve({ chunks, kept });
        else reject(new Error(`Paint's read of ${url}: ${data.error}`));
      }
    });
    return promise;
  }

  state(url) {
    return this.owed > 0 ? "being kept" : this.lost.has(url) ? "not kept" : "kept";
  }
}
