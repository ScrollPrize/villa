// The kernel's pages are 4 KiB at the least.
const PAGE = 4096;

self.onmessage = ({ data: { count, length } }) => {
  const buffers = Array.from({ length: count }, () => {
    const array = new Uint8Array(length);
    for (let at = 0; at < length; at += PAGE) array[at] = 0;
    return array.buffer;
  });
  self.postMessage({ length, buffers }, buffers);
};
