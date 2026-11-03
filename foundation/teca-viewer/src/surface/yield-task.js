export function yieldTask(signal = null) {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const { port1, port2 } = new MessageChannel();
    const cleanup = () => {
      port1.close();
      port2.close();
      signal?.removeEventListener("abort", aborted);
    };
    const aborted = () => {
      cleanup();
      reject(signal.reason);
    };
    port1.onmessage = () => {
      cleanup();
      resolve();
    };
    signal?.addEventListener("abort", aborted, { once: true });
    port2.postMessage(null);
  });
}
