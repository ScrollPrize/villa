import { lock, transaction } from "../../storage.js";

const STORE = "skeletons";
const DATABASE = { name: "teca-points", version: 1, upgrade: (database) => database.createObjectStore(STORE) };
const KEEP_DELAY_MS = 500;

async function readStored(url) {
  const [text] = await transaction(DATABASE, [STORE], "readonly", (store) => [store.get(url)]);
  return text ?? null;
}

async function writeStored(url, text) {
  await transaction(DATABASE, [STORE], "readwrite", (store) => {
    if (text === null) store.delete(url);
    else store.put(text, url);
  });
}

export class Keeping {
  constructor(failed, changed) {
    this.failed = failed;
    this.changed = changed;
    this.waiting = null;
    this.owed = 0;
    this.unkept = null;
    this.writes = Promise.resolve();
    this.held = null;
  }

  async hold(url, waiting) {
    if (this.held?.url !== url) {
      const before = this.held;
      if (before !== null) void this.writes.then(before.release);
      this.held = Object.assign(lock(`teca-points ${url}`), { url });
    }
    const held = this.held;
    if (!(await held.free) && !held.granted) await waiting();
    await held.ready;
  }

  holds(url) {
    return this.held?.url === url && this.held.granted;
  }

  soon(url, text) {
    window.clearTimeout(this.waiting?.timer);
    this.waiting = { timer: window.setTimeout(() => this.now(), KEEP_DELAY_MS), url, text };
  }

  now(url = this.waiting?.url, text = this.waiting?.text) {
    window.clearTimeout(this.waiting?.timer);
    this.waiting = null;
    if (url === undefined) return;
    const written = text();
    const held = this.holds(url);
    this.owed += 1;
    this.writes = this.writes
      .then(async () => {
        this.unkept = url;
        if (!held) throw new Error(`this page does not hold the nodes of ${url}, so it writes none of them.`);
        await writeStored(url, written);
        this.unkept = null;
      })
      .catch(this.failed)
      .finally(() => {
        this.owed -= 1;
        this.changed();
      });
  }

  state(url) {
    return this.waiting !== null || this.owed > 0 ? "being kept" : this.unkept === url ? "not kept" : "kept";
  }

  read(url) {
    return this.writes.then(() => readStored(url));
  }
}
