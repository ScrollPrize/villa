import { DOWNLOAD_VALIDATION_TIME_LIMIT_MS } from "../config.js";
import { freshUntil, storableEtag, strongEtag } from "./download-cache.js";
import { abortable, fetchWithRetry } from "./request-queue.js";

export class CachedDownload {
  constructor(group, cache) {
    this.group = group;
    this.cache = cache;
    this.versions = new Map();
    this.validations = new Map();
  }

  async fetch(url, options, requestUrl = url) {
    const signal = options.signal;
    signal?.throwIfAborted();
    const lease = await this.cache.lookup(this.group, url, options.headers?.Range ?? "");
    signal?.throwIfAborted();
    if (lease?.entry !== undefined) {
      const { entry } = lease;
      let version = this.versions.get(url);
      if (version === undefined && entry.freshUntil > Date.now()) version = this.remember(url, entry.etag, true);
      if (version === undefined) {
        let check = this.validations.get(url);
        if (check === undefined) {
          const stopped = new AbortController();
          const checking = AbortSignal.any([stopped.signal, AbortSignal.timeout(DOWNLOAD_VALIDATION_TIME_LIMIT_MS)]);
          check = { readers: 0, stopped };
          check.version = (async () => {
            let checked = null;
            let until = null;
            try {
              const validation = await fetchWithRetry(requestUrl, checking, { method: "HEAD", cache: "no-cache" });
              checked = validation.status === 200 ? storableEtag(validation.headers) : null;
              until = freshUntil(validation.headers, Date.now());
              await validation.body?.cancel();
            } catch (error) {
              if (stopped.signal.aborted) return null;
              console.warn(`Could not validate stored download; reading over HTTP: ${error.message}`);
            }
            return { ...this.remember(url, checked, checked !== null), until };
          })().finally(() => {
            if (this.validations.get(url) === check) this.validations.delete(url);
          });
          this.validations.set(url, check);
        }
        check.readers++;
        try {
          version = await abortable(check.version, signal);
        } finally {
          check.readers--;
          if (check.readers === 0 && version === undefined) {
            check.stopped.abort();
            if (this.validations.get(url) === check) this.validations.delete(url);
          }
        }
        signal?.throwIfAborted();
        if (version.reusable && version.etag === entry.etag && version.until > Date.now())
          void this.cache.refresh(lease, version.until);
      }
      const etag = version.reusable ? version.etag : null;
      if (etag !== null && etag === entry.etag) {
        signal?.throwIfAborted();
        this.cache.hitBytes += entry.bytes;
        return {
          response: new Response(entry.body, { status: entry.status, headers: entry.headers }),
          save: async () => {},
        };
      }
    }
    const response = await fetchWithRetry(requestUrl, signal, options);
    if (response.status === 200 || response.status === 206) {
      const etag = strongEtag(response.headers);
      const previous = this.versions.get(url)?.etag;
      if (previous != null && etag !== null && previous !== etag) {
        await response.body?.cancel();
        throw new Error("The source changed while open. Reopen it to read a consistent version.");
      }
      this.remember(url, etag, storableEtag(response.headers) !== null);
    }
    return {
      response,
      save: async (bytes) => {
        signal?.throwIfAborted();
        if (response.status === 200 || response.status === 206)
          await this.cache.put(lease, this.group, response, bytes);
        signal?.throwIfAborted();
      },
    };
  }

  remember(url, etag, reusable) {
    const previous = this.versions.get(url)?.etag;
    if (previous != null && etag !== null && previous !== etag)
      throw new Error("The source changed while open. Reopen it to read a consistent version.");
    const version = { etag: etag ?? previous ?? null, reusable };
    this.versions.set(url, version);
    return version;
  }
}
