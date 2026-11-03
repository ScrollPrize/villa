import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { test } from "node:test";

const ROOT = resolve(import.meta.dirname, "..");
const SRC = join(ROOT, "src");
const LAYERS = [
  "vendor",
  "config.js",
  "math.js",
  "planes.js",
  "zip.js",
  "storage.js",
  "element-rect.js",
  "worker-pool.js",
  "state.js",
  "volume",
  "gpu",
  "surface",
  "session.js",
  "views",
  "render",
  "ui",
  "debug",
  "extensions",
  "main.js",
  "startup.js",
];
const EXTENSION_IMPORTS = ["vendor", "math.js", "zip.js", "storage.js"];
const PAGE_FREE = ["volume", "surface"];
const PAGE =
  /(?<![\w$.-])(?:(?:document|window|navigator|location|history|globalThis)\.|(?:localStorage|sessionStorage|indexedDB|caches|requestAnimationFrame|cancelAnimationFrame|ResizeObserver)\b)|\.getBoundingClientRect\(|\.style\./;
const SPECIFIER = new RegExp(
  [
    String.raw`^(?:import|export)\s+(?:\*\s+as\s+\w+|\*|\w+|\{[^}]*\}|\w+\s*,\s*\{[^}]*\})\s*from\s*"([^"]+)"`,
    String.raw`^import\s*"([^"]+)"`,
    String.raw`\bimport\(\s*"([^"]+)"`,
    String.raw`\bnew URL\(\s*"([^"]+)",\s*import\.meta\.url\)`,
  ].join("|"),
  "gm",
);
const CLAUSE = /^(?:import|export)\s+(\*\s+as\s+\w+|\*|\w+|\{[^}]*\}|\w+\s*,\s*\{[^}]*\})\s*from\s*"([^"]+)"/gm;
const VOLUME_INTERNALS =
  /\.(?:axes|storageShape|storageChunks|storageFormat|chunkSeparator|fillValue|transformScale|transformTranslation|numericType)\b|\.metadata\.(?!physicalUnits_ZYX\b|displayEncoding\b)\w+/g;
const KEBAB = /^(?:[a-z]+\/|extensions\/[a-z0-9]+(?:-[a-z0-9]+)*\/)?[a-z0-9]+(?:-[a-z0-9]+)*(?:\.worker)?\.js$/;

const files = readdirSync(SRC, { recursive: true })
  .filter((name) => name.endsWith(".js"))
  .sort();
const part = (file) => (file.startsWith("..") ? "vendor" : file.split("/")[0]);
const extension = (file) => /^extensions\/([a-z0-9-]+)/.exec(file)?.[1].replace(/\.js$/, "") ?? null;
const edges = files.flatMap((file) => {
  const source = readFileSync(join(SRC, file), "utf8");
  return [...source.matchAll(SPECIFIER)].map((match) => {
    const specifier = match[1] ?? match[2] ?? match[3] ?? match[4];
    return {
      file,
      at: `src/${file}:${source.slice(0, match.index).split("\n").length}`,
      specifier,
      worker: match[4] !== undefined,
      target: relative(SRC, resolve(dirname(join(SRC, file)), specifier)),
    };
  });
});
const none = (found) => assert.ok(found.length === 0, `${found.length} to change:\n${found.join("\n")}`);
const lineOf = (text, pattern) => {
  const index = text.search(pattern);
  assert.notEqual(index, -1, `nothing matches ${pattern}`);
  return text.slice(0, index).split("\n").length;
};
const contract = readFileSync(join(ROOT, "docs/extensions.md"), "utf8");
const extensionCode = new Map();
for (const file of files.filter((name) => extension(name) !== null)) {
  const code = readFileSync(join(SRC, file), "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/.*$/gm, "$1");
  extensionCode.set(extension(file), `${extensionCode.get(extension(file)) ?? ""}\n${code}`);
}

test("every import is a relative path to a file that exists", () => {
  const found = [];
  for (const { at, specifier, target } of edges) {
    if (!/^\.\.?\//.test(specifier)) found.push(`${at} imports ${specifier}; name the file by a relative path`);
    if (!existsSync(join(SRC, target))) found.push(`${at} imports ${specifier}, which does not exist`);
    if (target.startsWith("..") && !target.startsWith("../vendor/"))
      found.push(`${at} imports ${specifier}, outside src/`);
  }
  none(found);
});

test("a worker is started from its URL and never imported", () => {
  none(
    edges
      .filter(({ specifier, worker }) => worker !== specifier.endsWith(".worker.js"))
      .map(({ at }) => `${at}: only a *.worker.js file is started by its URL`),
  );
});

test("file names are lower-case words joined by hyphens", () => {
  none(files.filter((file) => !KEBAB.test(file)).map((file) => `src/${file}: name it in kebab-case`));
});

test("every part has its place in LAYERS", () => {
  const parts = new Set(files.map(part));
  none([
    ...[...parts].filter((name) => !LAYERS.includes(name)).map((name) => `src/${name} is missing from LAYERS`),
    ...LAYERS.slice(1)
      .filter((name) => !parts.has(name))
      .map((name) => `LAYERS names ${name}, which src/ has not`),
  ]);
});

test("a part imports only itself and the parts below it", () => {
  none(
    edges
      .filter(({ file, target }) => LAYERS.indexOf(part(target)) > LAYERS.indexOf(part(file)))
      .map(({ file, at, target }) => `${at} imports ${part(target)} from ${part(file)}, which LAYERS puts above it`),
  );
});

test("main.js alone attaches an extension, and an extension imports its own files and EXTENSION_IMPORTS", () => {
  const found = [];
  for (const { file, at, target } of edges) {
    const from = extension(file);
    const to = extension(target);
    if (from === null && to !== null && file !== "main.js")
      found.push(`${at} imports extension ${to}; attach it in main.js`);
    if (from !== null && to !== from && !EXTENSION_IMPORTS.includes(part(target)))
      found.push(`${at} imports ${target} into extension ${from}; take it from the viewer main.js hands it`);
  }
  none(found);
});

test("main.js attaches each file of src/extensions/, and each has its browser check", () => {
  const attached = [...readFileSync(join(SRC, "main.js"), "utf8").matchAll(/extensions\.extend\("([a-z-]+)"/g)]
    .map((match) => match[1])
    .sort();
  const written = readdirSync(join(SRC, "extensions"))
    .filter((name) => name.endsWith(".js"))
    .map((name) => name.slice(0, -".js".length))
    .sort();
  assert.deepEqual(
    attached,
    written,
    "src/main.js: attach each file of src/extensions/; delete the files of one left out",
  );
  none(
    attached
      .filter((name) => !existsSync(join(ROOT, "tests", `${name}.e2e.py`)))
      .map((name) => `tests/${name}.e2e.py: ${name} has none`),
  );
});

test("each part EXTENSION_IMPORTS names is imported by an extension, and docs/extensions.md lists them", () => {
  const imported = new Set(
    edges
      .filter(({ file, target }) => extension(file) !== null && extension(target) === null)
      .map(({ target }) => part(target)),
  );
  const at = `tests/architecture.test.js:${lineOf(readFileSync(import.meta.filename, "utf8"), /^const EXTENSION_IMPORTS/m)}`;
  const list = /^- It imports its own files, (.*), nothing else of the core$/m;
  const listed = [...contract.match(list)[1].matchAll(/`([^`]+)`/g)].map((match) => match[1].replace(/\/$/, ""));
  none([
    ...EXTENSION_IMPORTS.filter((name) => !imported.has(name)).map(
      (name) => `${at} names ${name}, which no extension imports; take it out, and out of docs/extensions.md's list`,
    ),
    ...(listed.join() === EXTENSION_IMPORTS.join()
      ? []
      : [
          `docs/extensions.md:${lineOf(contract, list)} lists ${listed.join(", ")}; list ${EXTENSION_IMPORTS.join(", ")}, as ${at}`,
        ]),
  ]);
});

test("an extension's element ids start with its name", () => {
  const found = [];
  for (const file of files.filter((name) => extension(name) !== null)) {
    const source = readFileSync(join(SRC, file), "utf8");
    for (const match of source.matchAll(/\bid="([^"$]+)"|\.id\s*=\s*["'`]([^"'`$]+)/g)) {
      const id = match[1] ?? match[2];
      const line = source.slice(0, match.index).split("\n").length;
      if (!id.startsWith(`${extension(file)}-`))
        found.push(`src/${file}:${line} names element ${id}; start it with ${extension(file)}-`);
    }
  }
  none(found);
});

test("an extension reads of a volume only what docs/extensions.md's contract names", () => {
  const found = [];
  for (const file of files.filter((name) => extension(name) !== null)) {
    const source = readFileSync(join(SRC, file), "utf8");
    for (const match of source.matchAll(VOLUME_INTERNALS)) {
      const line = source.slice(0, match.index).split("\n").length;
      found.push(
        `src/${file}:${line} reads ${match[0]} of a volume; docs/extensions.md's openVolume names what it may`,
      );
    }
  }
  none(found);
});

test("an extension's classes are the side panel's that docs/extensions.md lists, or its own, styled in its panel", () => {
  const styles = readFileSync(join(ROOT, "styles.css"), "utf8");
  const shared = [
    ...contract.match(/Its panel may take the side panel's classes: ([^.]*)\./)[1].matchAll(/`([\w-]+)`/g),
  ].map((match) => match[1]);
  const found = shared
    .filter((name) => !new RegExp(`\\.${name}(?![\\w-])`).test(styles))
    .map((name) => `docs/extensions.md lists class ${name}, which styles.css does not style`);
  for (const name of readdirSync(join(SRC, "extensions")).filter((entry) => entry.endsWith(".js")))
    if (styles.includes(`.${name.slice(0, -".js".length)}-`))
      found.push(`styles.css styles .${name.slice(0, -".js".length)}-; move it to a <style> in that extension's panel`);
  for (const file of files.filter((name) => extension(name) !== null)) {
    const source = readFileSync(join(SRC, file), "utf8");
    for (const match of source.matchAll(
      /\bclass="([^"$]+)"|\.className\s*=\s*"([^"]+)"|\.classList\.\w+\("([^"]+)"/g,
    )) {
      const line = source.slice(0, match.index).split("\n").length;
      for (const name of (match[1] ?? match[2] ?? match[3]).split(/\s+/))
        if (!shared.includes(name) && !name.startsWith(`${extension(file)}-`))
          found.push(
            `src/${file}:${line} takes class ${name}; start it with ${extension(file)}- or list it in docs/extensions.md`,
          );
    }
  }
  none(found);
});

test("each member of the viewer the extensions are handed is used by one", () => {
  const members = /const viewer = \{([^}]*)\}/
    .exec(readFileSync(join(SRC, "ui/extensions.js"), "utf8"))[1]
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
  const extensions = files
    .filter((file) => extension(file) !== null)
    .map((file) => readFileSync(join(SRC, file), "utf8"))
    .join("\n");
  none(
    members
      .filter((member) => !new RegExp(`\\bviewer\\.${member}\\b`).test(extensions))
      .map((member) => `src/ui/extensions.js hands the extensions ${member}, which none uses; delete it and its seam`),
  );
});

test("no file imports itself through others", () => {
  const next = new Map(files.map((file) => [file, []]));
  for (const { file, target, worker } of edges) if (!worker && !target.startsWith("..")) next.get(file).push(target);
  const done = new Set();
  const found = [];
  const visit = (file, path) => {
    const loop = path.indexOf(file);
    if (loop !== -1) {
      found.push(`import cycle: ${[...path.slice(loop), file].join(" -> ")}`);
      return;
    }
    if (done.has(file)) return;
    for (const target of next.get(file)) visit(target, [...path, file]);
    done.add(file);
  };
  for (const file of files) visit(file, []);
  none(found);
});

test("volume/ and surface/ touch no page object", () => {
  const found = [];
  for (const file of files.filter((name) => PAGE_FREE.includes(part(name)))) {
    readFileSync(join(SRC, file), "utf8")
      .split("\n")
      .forEach((text, i) => {
        if (/^\s*(?:\/\/|\/?\*)/.test(text)) return;
        const match = PAGE.exec(text);
        if (match !== null) found.push(`src/${file}:${i + 1} reads ${match[0]}; take it from main.js as an argument`);
      });
  }
  none(found);
});

test("a file exports only what another file or a test imports", () => {
  const tests = readdirSync(join(ROOT, "tests"))
    .filter((name) => name.endsWith(".js"))
    .map((name) => join("..", "tests", name));
  const imported = new Map(files.map((file) => [file, new Set()]));
  for (const file of [...files, ...tests]) {
    for (const [, clause, specifier] of readFileSync(join(SRC, file), "utf8").matchAll(CLAUSE)) {
      const names = imported.get(relative(SRC, resolve(dirname(join(SRC, file)), specifier)));
      if (names === undefined) continue;
      if (clause.startsWith("*")) names.add("*");
      if (/^\w/.test(clause)) names.add("default");
      for (const name of (/\{([^}]*)\}/.exec(clause)?.[1] ?? "").split(","))
        names.add(name.trim().split(/\s+as\s+/)[0]);
    }
  }
  const found = [];
  for (const file of files) {
    const source = readFileSync(join(SRC, file), "utf8");
    const names = imported.get(file);
    const exported = [
      ...[...source.matchAll(/^export\s+(?:async\s+)?(?:function\*?|class|const|let)\s+(\w+)/gm)].map(
        (match) => match[1],
      ),
      ...[...source.matchAll(/^export\s*\{([^}]*)\}/gm)].flatMap((match) =>
        match[1].split(",").map((name) =>
          name
            .trim()
            .split(/\s+as\s+/)
            .pop(),
        ),
      ),
      ...(/^export default/m.test(source) ? ["default"] : []),
    ];
    for (const name of exported.filter(Boolean)) {
      if (!names.has(name) && !names.has("*")) found.push(`src/${file} exports ${name}, which no file or test imports`);
    }
  }
  none(found);
});

test("a method is called from somewhere", () => {
  const corpus = ["src", "tests", "tools"]
    .flatMap((dir) =>
      readdirSync(join(ROOT, dir), { recursive: true })
        .filter((name) => /\.(js|py)$/.test(name))
        .map((name) => {
          const text = readFileSync(join(ROOT, dir, name), "utf8");
          if (name.endsWith(".py")) return text;
          return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
        }),
    )
    .join("\n");
  const called = new Set([...corpus.matchAll(/[.?](\w+)/g)].map((match) => match[1]));
  for (const [, names] of corpus.matchAll(/\{([^{}]*)\}\s*=[^=>]/g))
    for (const name of names.split(",")) called.add(name.split(":")[0].trim());
  const METHOD = /^ {2,}(?:(?:async|get|set|static)\s+)*(\w+)\s*\([^)]*\)\s*\{$/gm;
  const found = [];
  for (const file of files) {
    const source = readFileSync(join(SRC, file), "utf8");
    for (const match of source.matchAll(METHOD)) {
      if (["if", "for", "while", "switch", "catch", "constructor"].includes(match[1]) || called.has(match[1])) continue;
      const line = source.slice(0, match.index).split("\n").length;
      found.push(`src/${file}:${line} defines ${match[1]}, which nothing calls; delete it`);
    }
  }
  none(found);
});

test("no file imports by a computed name", () => {
  none(
    files
      .filter((file) => /\bimport\(\s*[^"\s]/.test(readFileSync(join(SRC, file), "utf8")))
      .map((file) => `src/${file} imports by a computed name; import the file by its path`),
  );
});
