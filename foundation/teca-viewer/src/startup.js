(() => {
  const message = document.querySelector("#startup-message");
  const title = document.querySelector("#startup-title");
  const detail = document.querySelector("#startup-detail");
  const app = document.querySelector("#app");
  if (location.protocol === "file:") {
    title.textContent = "Open the viewer through localhost or HTTPS";
    detail.textContent = "The browser cannot load this viewer's JavaScript modules from a file:// address.";
    document.querySelector("#startup-local").hidden = false;
    return;
  }
  const unavailable = [];
  if (!globalThis.isSecureContext) unavailable.push("a secure HTTPS or localhost address");
  if (typeof Worker !== "function") unavailable.push("Web Workers");
  if (typeof WebAssembly !== "object") unavailable.push("WebAssembly");
  if (!globalThis.crypto?.subtle) unavailable.push("browser cryptography");
  try {
    const key = "viewer-storage-check";
    localStorage.setItem(key, "1");
    localStorage.removeItem(key);
  } catch (error) {
    unavailable.push(`browser storage (allow site data for this address: ${error.message})`);
  }
  if (unavailable.length > 0) {
    title.textContent = "Browser requirements are unavailable";
    detail.textContent = `This viewer requires ${unavailable.join(", ")}.`;
    return;
  }
  const gl = document.createElement("canvas").getContext("webgl2");
  if (gl === null) {
    title.textContent = "WebGL 2 is unavailable";
    detail.textContent = "Enable graphics acceleration in your browser and restart it, then reload this page.";
    return;
  }
  gl.getExtension("WEBGL_lose_context")?.loseContext();
  performance.mark("viewer:startup");
  detail.textContent = "Loading viewer modules…";
  import("./main.js").catch((error) => {
    app.inert = true;
    title.textContent = "The viewer could not start";
    detail.textContent = `${error.message} Check the browser console for details, then reload.`;
    message.close();
    message.showModal();
    console.error("Viewer startup failed", error);
  });
})();
