// [bucket.]s3.[dualstack.]<region>.amazonaws.com on its default port.
// The global bucket.s3.amazonaws.com names no region and does not match.
const S3_HOST = /^(?:([a-z0-9-]+)\.)?s3\.(?:dualstack\.)?([a-z]{2}(?:-[a-z]+)+-\d)\.amazonaws\.com$/;

const FNV_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;
const FNV_FOLD = 13;

export function mirrors(url) {
  const { protocol, host, origin, pathname, search } = new URL(url);
  const match = S3_HOST.exec(host);
  // A query may carry a signature, which covers the hostname.
  if (match === null || search !== "") return [{ url, origin }];
  const region = match[2];
  const bucket = match[1] === undefined ? pathname.split("/")[1] : match[1];
  // A dotted bucket has no certificate as a hostname.
  if (!/^[a-z0-9-]+$/.test(bucket)) return [{ url, origin }];
  const path = match[1] === undefined ? pathname.slice(bucket.length + 1) : pathname;
  return ["", "dualstack."].flatMap((stack) => {
    const hosted = `${protocol}//${bucket}.s3.${stack}${region}.amazonaws.com`;
    const shared = `${protocol}//s3.${stack}${region}.amazonaws.com`;
    return [
      { url: hosted + path, origin: hosted },
      { url: `${shared}/${bucket}${path}`, origin: shared },
    ];
  });
}

// The mirror is fixed per object, so HTTP and disk caches keep one key for it. A plain sum of character codes would
// send every chunk of a plane z + y + x = constant to one hostname.
export function mirror(url) {
  const all = mirrors(url);
  if (all.length === 1) return all[0];
  const name = all[0].url;
  let hash = FNV_BASIS;
  for (let index = 0; index < name.length; index++) {
    hash = Math.imul(hash ^ name.charCodeAt(index), FNV_PRIME);
    hash ^= hash >>> FNV_FOLD;
  }
  return all[(hash >>> 0) % all.length];
}
