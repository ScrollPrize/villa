const SLICE_MS = 8;

// Resolves in a task of its own, through a message: a timeout waits 4 ms once nested.
function nextTask() {
  const { port1, port2 } = new MessageChannel();
  return new Promise((resolve) => {
    port1.onmessage = () => {
      port1.close();
      resolve();
    };
    port2.postMessage(null);
  });
}

export function slices() {
  let since = performance.now();
  return async (drawn) => {
    if (performance.now() - since < SLICE_MS) return;
    drawn?.();
    await nextTask();
    since = performance.now();
  };
}
