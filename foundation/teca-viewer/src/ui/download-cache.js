import { DOWNLOAD_CACHE_GIB_RANGE } from "../config.js";
import { formatBytes } from "./format-bytes.js";

const GIB = 1024 ** 3;

// Nothing is read at boot: a page left while it creates the database waits in the back-forward cache with the creation
// unfinished, and every later page's open of the database waits behind it.
export function initDownloadCacheControls(elements, downloadStorage) {
  const {
    downloadCacheEnabled: enabled,
    downloadCacheBudget: budget,
    downloadCacheStatus: status,
    downloadCacheSources: sources,
    downloadCacheApply: apply,
    downloadCacheClear: clear,
    downloadCacheRefresh: refresh,
  } = elements;

  const show = async (restoreControls = false) => {
    const state = await downloadStorage.status();
    if (restoreControls) {
      enabled.checked = state.enabled;
      budget.value = String(state.byteLimit / GIB);
    }
    status.textContent =
      state.error === null
        ? `${formatBytes(state.used)} / ${formatBytes(state.byteLimit)} stored, ${formatBytes(state.hitBytes)} reused this page load. ${state.enabled ? "Saving downloads." : "Saving is off."}`
        : `Disk cache paused: ${state.error}. Downloads continue over HTTP. Clear storage or apply settings to retry.`;
    sources.replaceChildren();
    for (const group of state.groups) {
      const row = document.createElement("div");
      row.className = "readout-line download-source";
      const label = document.createElement("span");
      const url = new URL(group.id);
      label.textContent = decodeURIComponent(url.pathname).split("/").filter(Boolean).slice(-3).join(" / ");
      label.title = group.id;
      const size = document.createElement("span");
      size.textContent = formatBytes(group.bytes);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.setAttribute("aria-label", `Remove downloaded data for ${label.textContent}`);
      remove.addEventListener("click", () => run(remove, () => downloadStorage.clear(group.id)));
      row.append(label, size, remove);
      sources.append(row);
    }
  };
  const run = async (button, action) => {
    button.disabled = true;
    try {
      await action();
      await show();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  };
  apply.addEventListener("click", () =>
    run(apply, async () => {
      const gib = Number(budget.value);
      if (!Number.isFinite(gib) || gib < DOWNLOAD_CACHE_GIB_RANGE[0] || gib > DOWNLOAD_CACHE_GIB_RANGE[1])
        throw new Error(`Choose a disk budget from ${DOWNLOAD_CACHE_GIB_RANGE.join(" to ")} GiB.`);
      await downloadStorage.configure(enabled.checked, Math.round(gib * GIB));
    }),
  );
  clear.addEventListener("click", () => run(clear, () => downloadStorage.clear()));
  refresh.addEventListener("click", () => run(refresh, async () => {}));
  if (!downloadStorage.available) {
    for (const control of [enabled, budget, apply, clear, refresh]) control.disabled = true;
    status.textContent = "Download storage is unavailable in this browser.";
    return { refresh: () => {} };
  }
  return {
    refresh: () =>
      show(true).catch((error) => {
        status.textContent = `Download storage unavailable: ${error.message}`;
      }),
  };
}
