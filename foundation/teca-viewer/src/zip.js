// Neither zip64 nor a split zip is supported.

import { ZIP_BLOB_BYTES } from "./config.js";

const ZIP_MAX_ENTRIES = 65535;
const ZIP_MAX_COMMENT = 65535;
const ZIP_MAX_BYTES = 0xffffffff;
const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, index) => {
  let crc = index;
  for (let bit = 0; bit < 8; bit++) crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  return crc;
});
// A CRC takes 8 bytes a step: one byte a step took 21 ms for the 5.0 MB zip of a stroke of radius 256 across the
// Surface, in the page.
const CRC_TABLES = Uint32Array.from({ length: 8 * 256 }, (_, index) => {
  let crc = CRC_TABLE[index % 256];
  for (let k = 0; k < Math.floor(index / 256); k++) crc = CRC_TABLE[crc & 0xff] ^ (crc >>> 8);
  return crc;
});

function crc32(bytes) {
  let crc = 0xffffffff;
  let index = 0;
  for (const end = bytes.length - (bytes.length % 8); index < end; index += 8) {
    const low = crc ^ (bytes[index] | (bytes[index + 1] << 8) | (bytes[index + 2] << 16) | (bytes[index + 3] << 24));
    const high = bytes[index + 4] | (bytes[index + 5] << 8) | (bytes[index + 6] << 16) | (bytes[index + 7] << 24);
    crc =
      CRC_TABLES[7 * 256 + (low & 0xff)] ^
      CRC_TABLES[6 * 256 + ((low >>> 8) & 0xff)] ^
      CRC_TABLES[5 * 256 + ((low >>> 16) & 0xff)] ^
      CRC_TABLES[4 * 256 + (low >>> 24)] ^
      CRC_TABLES[3 * 256 + (high & 0xff)] ^
      CRC_TABLES[2 * 256 + ((high >>> 8) & 0xff)] ^
      CRC_TABLES[256 + ((high >>> 16) & 0xff)] ^
      CRC_TABLES[high >>> 24];
  }
  for (; index < bytes.length; index++) crc = CRC_TABLE[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

// A forged large size does not allocate its output in advance.
async function inflated(stored, count, name) {
  const reader = new Blob([stored]).stream().pipeThrough(new DecompressionStream("deflate-raw")).getReader();
  const chunks = [];
  let length = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value.length > count - length) throw new Error(`The zip's ${name} expands past its declared ${count} bytes.`);
      chunks.push(value);
      length += value.length;
    }
  } catch (error) {
    await reader.cancel(error);
    throw error;
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(length);
  let at = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, at);
    at += chunk.length;
  }
  return bytes;
}

export async function zipped(entries, pause, deflate) {
  const encoder = new TextEncoder();
  const blobs = [];
  let taken = [];
  const central = [];
  let offset = 0;
  let held = 0;
  let count = 0;
  for (const { name, bytes } of entries) {
    await pause();
    count += 1;
    if (count > ZIP_MAX_ENTRIES) throw new Error(`More than ${ZIP_MAX_ENTRIES} files; a zip lists at most that many.`);
    const path = encoder.encode(name);
    // Through a Blob's stream the page deflates 6.1 MB of NML in 48 to 68 ms holding no frame past 33 ms, where one
    // write of it into the stream held frames 33 to 50 ms; 61 MB holds the page 0.39 s either way.
    const data = deflate
      ? new Uint8Array(
          await new Response(
            new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate-raw")),
          ).arrayBuffer(),
        )
      : bytes;
    const common = new DataView(new ArrayBuffer(26));
    common.setUint16(0, 20, true);
    common.setUint16(2, 0x0800, true);
    common.setUint16(4, deflate ? 8 : 0, true);
    common.setUint16(8, 0x21, true);
    common.setUint32(10, crc32(bytes), true);
    common.setUint32(14, data.length, true);
    common.setUint32(18, bytes.length, true);
    common.setUint16(22, path.length, true);
    // Each header is one array, as a Blob of thousands of small parts is slow to make: with four parts an entry, that
    // stroke's zip took 40 to 50 ms in tasks of up to 17.2 ms, and with two 30 to 43 ms in tasks of up to 16.4 ms.
    const local = new Uint8Array(30 + path.length);
    local.set([0x50, 0x4b, 3, 4]);
    local.set(new Uint8Array(common.buffer), 4);
    local.set(path, 30);
    const record = new Uint8Array(46 + path.length);
    record.set([0x50, 0x4b, 1, 2, 20, 0]);
    record.set(new Uint8Array(common.buffer), 6);
    new DataView(record.buffer).setUint32(42, offset, true);
    record.set(path, 46);
    taken.push(local, data);
    central.push(record);
    offset += local.length + data.length;
    if (offset - held >= ZIP_BLOB_BYTES) {
      blobs.push(new Blob(taken));
      taken = [];
      held = offset;
    }
  }
  const directory = new Uint8Array(central.reduce((total, record) => total + record.length, 0));
  if (offset + directory.length > ZIP_MAX_BYTES)
    throw new Error(`${offset + directory.length} bytes of files and directory; a zip holds at most ${ZIP_MAX_BYTES}.`);
  let at = 0;
  for (const record of central) {
    directory.set(record, at);
    at += record.length;
  }
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, count, true);
  end.setUint16(10, count, true);
  end.setUint32(12, directory.length, true);
  end.setUint32(16, offset, true);
  return new Blob([...blobs, ...taken, directory, new Uint8Array(end.buffer)], { type: "application/zip" });
}

export async function unzipped(zip) {
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  let end = zip.length - 22;
  const first = Math.max(0, end - ZIP_MAX_COMMENT);
  while (
    end >= first &&
    (view.getUint32(end, true) !== 0x06054b50 || end + 22 + view.getUint16(end + 20, true) !== zip.length)
  )
    end--;
  if (end < first) throw new Error("The file is not a zip: it has no end of central directory.");
  const count = view.getUint16(end + 10, true);
  if (
    view.getUint16(end + 4, true) !== 0 ||
    view.getUint16(end + 6, true) !== 0 ||
    view.getUint16(end + 8, true) !== count
  )
    throw new Error("The zip is split across disks.");
  const decoder = new TextDecoder();
  const entries = [];
  const names = new Set();
  let at = view.getUint32(end + 16, true);
  const directory = at;
  if (at + view.getUint32(end + 12, true) !== end) throw new Error("The zip's central directory size is wrong.");
  for (let index = 0; index < count; index++) {
    if (at + 46 > end || view.getUint32(at, true) !== 0x02014b50)
      throw new Error(`The zip has no central header at ${at}.`);
    const method = view.getUint16(at + 10, true);
    const size = view.getUint32(at + 20, true);
    const nameLength = view.getUint16(at + 28, true);
    const header = view.getUint32(at + 42, true);
    const next = at + 46 + nameLength + view.getUint16(at + 30, true) + view.getUint16(at + 32, true);
    if (next > end || header + 30 > directory || view.getUint32(header, true) !== 0x04034b50)
      throw new Error(`The zip has a broken header at ${header}.`);
    const name = decoder.decode(zip.subarray(at + 46, at + 46 + nameLength));
    const start = header + 30 + view.getUint16(header + 26, true) + view.getUint16(header + 28, true);
    if (start + size > directory) throw new Error(`The zip's ${name} is cut short.`);
    if (names.has(name)) throw new Error(`The zip has ${name} twice.`);
    names.add(name);
    if (view.getUint16(at + 8, true) & 1) throw new Error(`The zip's ${name} is encrypted.`);
    if (method !== 0 && method !== 8) throw new Error(`The zip's ${name} is compressed with method ${method}.`);
    const stored = zip.subarray(start, start + size);
    const bytes = method === 0 ? stored : await inflated(stored, view.getUint32(at + 24, true), name);
    if (bytes.length !== view.getUint32(at + 24, true) || crc32(bytes) !== view.getUint32(at + 16, true))
      throw new Error(`The zip's ${name} has the wrong size or CRC.`);
    entries.push({ name, bytes });
    at = next;
  }
  if (at !== end) throw new Error(`The zip's central directory runs on past ${at}.`);
  return entries;
}
