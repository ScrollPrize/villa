import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { test } from "node:test";

const ROOT = resolve(import.meta.dirname, "..");
const OWNERS = {
  "ui/camera.js": [
    "activeViewport",
    "fullscreenViewport",
    "plane",
    "center_ZYX",
    "zoom",
    "rotationMode",
    "rotation",
    "paneRotations",
    "cameraOrientation",
    "heldSliceStep",
    "crosshairVisible",
  ],
  "ui/display.js": ["displayBlack", "displayWhite", "displayGamma", "alone"],
  "ui/temporary-camera.js": ["temporaryCamera"],
  "volume/active-volume.js": [
    "sources",
    "activeSource",
    "volume",
    "selectedChannel",
    "manualScaleIndex",
    "manualCoarsestScaleIndex",
    "workGeneration",
    "contentGeneration",
  ],
  "surface/controller.js": ["surfaceView"],
  "surface/overlay-store.js": ["overlays", "overlayLineWidth"],
  "render/scheduler.js": ["renderToken", "dirtyPlanes", "lastInteractionDirtyPlanes"],
  "ui/held-navigation.js": ["heldSliceDirection", "heldZoomDirection", "heldRotation"],
  "ui/pointer.js": ["pointerDrag", "lastPointerClient", "tool", "panHeld"],
  "ui/extensions.js": ["extensions"],
  "ui/shell-panels.js": ["tabs"],
};
const PART_FIELDS = {
  "views/surface-pane.js": ["compareFlat", "interactionX"],
  "surface/controller.js": ["normalLocked"],
  "extensions/paint/holding.js": ["labels", "busy", "restoring"],
};
const REACH = [
  [/(?<!this)\.controls\./, "take the panel's functions from main.js"],
  [/\.output\.owner\b/, "take the renderer as rendering from the caller"],
  [/\.presentation\??\.resources\b/, "ask the SurfacePresentation, or take what it passes"],
  [/\.slicedPlane\.panes\b|\.levelCache\b(?<!this\.levelCache)/, "ask SlicedPlane"],
];
const MUTATORS = "push|splice|pop|shift|unshift|sort|reverse|fill|set|delete|clear";
const LINK = String.raw`(?:\.\w+\((?:[^()]|\([^()]*\))*\)|\.\w+|\[[^\]]+\])`;
const ASSIGN = String.raw`\s*(?:[-+*/|&]?=(?!=)|\+\+|--)`;
const WRITE = new RegExp(
  String.raw`(?:\+\+|--)\s*(?:this\.)?state\.(\w+)|\b(?:this\.)?state\.(\w+)${LINK}*` +
    String.raw`(?:${ASSIGN}|\.(?:${MUTATORS})\()|Object\.assign\(\s*(?:this\.)?state\.(\w+)`,
  "g",
);
const ALIAS = new RegExp(String.raw`(?:const|let)\s+(\w+)\s*=\s*(?:this\.)?state\.(\w+)${LINK}*\s*;`, "g");
const TAKEN = new RegExp(String.raw`(?:const|let)\s*\{([^}]*)\}\s*=\s*(?:this\.)?state(?:\.(\w+)${LINK}*)?\s*;`, "g");
const HISTORY = /\b(previously|formerly|used to be|new in this|legacy)\b/i;
const BANDAGE = /\b(workaround|hack|for now|temporary|temporarily|TODO|FIXME|XXX)\b/i;
const VAGUE = /\b\w*(?:Manager|Service|Handler|Impl)\b|\b\w+_(?:v2|fast)\b/;
// A worker imports the same modules, so main.js could not hand them in. Zarrita's codec registry is vendored, and its
// arrays take their decoders from it alone, so zarr-codecs.js wraps them there.
const SHARED = new Set([
  "element-rect.js rectangles",
  "surface/scale-stack.js brickScales",
  "volume/download-stats.js responseDownloads",
  "volume/request-queue.js originQueues",
  "volume/zarr-codecs.js decodePool",
  "volume/zarr-codecs.js registry",
]);
const LINE_LIMIT = 500;
const LONG_FILES = { "surface/controller.js": 648 };
const RETRIED = ["aiohttp.ClientConnectorError"];
const UNPAIRED_SLIDERS = [
  "pos-z",
  "pos-y",
  "pos-x",
  "src/extensions/layers.js opacity",
  "src/extensions/ink.js opacity",
];
const REPLACED = /\b(?:window|globalThis|navigator|document|\w+\.prototype)\.(?!__)\w+\s*=(?!=)/;
const PAGE = /\bglobalThis\b|\b(?:document|navigator|localStorage|sessionStorage|indexedDB)\./;

const read = (file) => readFileSync(join(ROOT, file), "utf8");
const sources = readdirSync(join(ROOT, "src"), { recursive: true })
  .filter((name) => name.endsWith(".js"))
  .sort();
const lines = (file) =>
  read(`src/${file}`)
    .split("\n")
    .map((text, i) => ({ text, at: `src/${file}:${i + 1}` }));
const units = readdirSync(join(ROOT, "tests"))
  .filter((name) => name.endsWith(".test.js"))
  .map((name) => `tests/${name}`);
const tools = readdirSync(join(ROOT, "tools"))
  .filter((name) => name.endsWith(".py"))
  .map((name) => `tools/${name}`);
const checks = readdirSync(join(ROOT, "tests"))
  .filter((name) => name.endsWith(".e2e.py"))
  .map((name) => `tests/${name}`);
const extensions = readdirSync(join(ROOT, "src/extensions"))
  .filter((name) => name.endsWith(".js"))
  .map((name) => name.slice(0, -".js".length));
const owns = (name, check) => check === `tests/${name}.e2e.py` || check.startsWith(`tests/${name}-`);
const none = (found) => assert.ok(found.length === 0, `${found.length} to change:\n${found.join("\n")}`);

test("every field of the viewer state is declared, and one file writes it, directly or through a local", () => {
  const declared = new Set([
    ...[...read("src/state.js").matchAll(/^ {4}(\w+):/gm)].map((match) => match[1]),
    ...[
      ...read("src/session.js")
        .match(/SESSION_KEYS = \[([^\]]*)\]/)[1]
        .matchAll(/"(\w+)"/g),
    ].map((match) => match[1]),
  ]);
  const owner = new Map(Object.entries(OWNERS).flatMap(([file, fields]) => fields.map((field) => [field, file])));
  const found = [];
  for (const field of owner.keys())
    if (!declared.has(field)) found.push(`OWNERS names ${field}, which state.js and SESSION_KEYS do not declare`);
  for (const file of sources) {
    const source = read(`src/${file}`);
    const aliases = [
      ...[...source.matchAll(ALIAS)].map(([, name, field]) => [name, field]),
      ...[...source.matchAll(TAKEN)].flatMap(([, names, field]) =>
        names.split(",").map((entry) => {
          const [key, name = key] = entry.split("=")[0].split(":");
          return [name.trim(), field ?? key.trim()];
        }),
      ),
    ].filter(([name, field]) => name !== "" && owner.get(field) !== file);
    const through = aliases.map(([name, field]) => [
      new RegExp(String.raw`(?<![.\w$])${name}(?:${LINK}+${ASSIGN}|${LINK}*\.(?:${MUTATORS})\()`),
      name,
      field,
    ]);
    for (const { text, at } of lines(file)) {
      if (/\bstate\[[^\]]+\]\s*=[^=]/.test(text)) found.push(`${at} writes a state field by a computed name`);
      if (/Object\.assign\(\s*(?:this\.)?state\s*,/.test(text)) found.push(`${at} assigns into the whole state`);
      for (const match of text.matchAll(WRITE)) {
        const field = match[1] ?? match[2] ?? match[3];
        if (!declared.has(field)) found.push(`${at} writes state.${field}, which state.js and SESSION_KEYS lack`);
        else if (file !== owner.get(field)) found.push(`${at} writes state.${field}, which ${owner.get(field)} owns`);
      }
      for (const [write, name, field] of through) {
        if (write.test(text)) found.push(`${at} writes state.${field} through ${name}, which ${owner.get(field)} owns`);
      }
    }
  }
  none(found);
});

test("a part's field that other parts read is written in its own file", () => {
  const found = [];
  for (const [owner, fields] of Object.entries(PART_FIELDS)) {
    const write = new RegExp(String.raw`\.(?:${fields.join("|")})\s*=(?!=)`);
    for (const file of sources.filter((name) => name !== owner)) {
      for (const { text, at } of lines(file)) {
        if (write.test(text) && !/\bstate\./.test(text))
          found.push(`${at} writes a field of ${owner}; call its method`);
      }
    }
  }
  none(found);
});

test("a skeleton of Points is changed in extensions/points/skeleton.js alone", () => {
  const others = sources.filter((name) => name !== "extensions/points/skeleton.js").flatMap(lines);
  none(
    others
      .filter(({ text }) => /\bskeleton\.\w+\s*=(?!=)/.test(text))
      .map(({ at }) => `${at} changes a skeleton; call a function of extensions/points/skeleton.js`),
  );
});

test("a part asks the parts it holds, not the parts they hold", () => {
  const found = [];
  for (const file of sources) {
    for (const { text, at } of lines(file)) {
      for (const [chain, instead] of REACH)
        if (chain.test(text)) found.push(`${at} reaches through a part; ${instead}`);
    }
  }
  none(found);
});

test("parts call each other: no DOM events between them", () => {
  const found = [];
  for (const file of sources) {
    for (const { text, at } of lines(file)) {
      if (/\b(CustomEvent|dispatchEvent)\b/.test(text)) found.push(`${at} sends an event; call the part instead`);
    }
  }
  none(found);
});

test("main.js hands each part named functions, and five reach a part built after them", () => {
  const FORWARD = [
    "surfaceViewDrawn -> extensions",
    "activate -> sources",
    "settled -> extensions",
    "setViewports -> camera",
    "surfacesChanged -> extensions",
  ];
  const all = lines("main.js").filter(({ text }) => !/^\s*(?:import|\/\/|\*|\/\*)/.test(text));
  const built = new Map();
  all.forEach(({ text }, i) => {
    const match = /^(?:const|let)\s+(?:\{([^}]*)\}|(\w+))\s*=/.exec(text);
    for (const name of (match?.[1] ?? match?.[2] ?? "").split(",")) built.set(name.split(":").at(-1).trim(), i);
  });
  const found = [];
  const forward = [];
  all.forEach(({ text, at }, i) => {
    if (/\.\.\.\w/.test(text)) found.push(`${at} spreads; name each function the part calls`);
    for (const [, name] of text.matchAll(/(?<![.\w])(\w+)(?![\w:])/g)) {
      if (built.get(name) > i) forward.push(`${text.trim().split(":")[0]} -> ${name}`);
    }
  });
  none(found);
  assert.deepEqual(forward, FORWARD, "src/main.js: build a part before the parts that call it");
});

test("a part uses every named argument it takes", () => {
  const PARAMETERS = /(?:function\s+\w+|constructor)\s*\(\s*\{([^}]*)\}\s*(?:=\s*\{\s*\})?\s*\)\s*\{/g;
  const found = [];
  for (const file of sources) {
    const source = read(`src/${file}`);
    for (const match of source.matchAll(PARAMETERS)) {
      let depth = 1;
      let end = match.index + match[0].length;
      for (; depth > 0; end++) depth += source[end] === "{" ? 1 : source[end] === "}" ? -1 : 0;
      const body = source.slice(match.index + match[0].length, end);
      const line = source.slice(0, match.index).split("\n").length;
      for (const name of match[1].split(",").map((entry) => entry.split("=")[0].split(":").at(-1).trim())) {
        if (name !== "" && !new RegExp(`\\b${name}\\b`).test(body))
          found.push(`src/${file}:${line} takes ${name} and never uses it`);
      }
    }
  }
  none(found);
});

test("the address has one writer", () => {
  const found = [];
  for (const file of sources.filter((name) => name !== "session.js")) {
    for (const { text, at } of lines(file)) {
      if (/history\.(replaceState|pushState)|location\.(href|search|hash)\s*=[^=]/.test(text))
        found.push(`${at} writes the address; call publishSession from session.js`);
    }
  }
  none(found);
});

test("a chunk key is built and read in gpu/addresses.js alone", () => {
  const found = [];
  for (const file of sources.filter((name) => name !== "gpu/addresses.js")) {
    for (const { text, at } of lines(file)) {
      if (/\.split\("\\0"\)|\.split\("\/"\)\.map\(Number\)|\.path\}(?:\\0|\/)\$\{[\w.]*chunk_ZYX\.join/.test(text))
        found.push(`${at} builds or splits a chunk key; call chunkKey or chunkKeyParts in gpu/addresses.js`);
    }
  }
  none(found);
});

test("a module keeps no state of its own outside main.js, the workers and SHARED", () => {
  const state = /^(?:export )?(?:let |const (\w+) = (?:new (?!Text(?:En|De)coder\()\w+\(\)|create\w+\())/;
  const vendored = (file) =>
    [...read(`src/${file}`).matchAll(/^import (?:\* as (\w+)|\{([^}]*)\}) from "[./]*vendor\//gm)].flatMap(
      ([, all, some]) => (all ? [all] : some.split(",").map((name) => name.split(" as ").pop().trim())),
    );
  const found = [];
  for (const file of sources.filter((name) => name !== "main.js" && !name.endsWith(".worker.js"))) {
    const names = vendored(file).join("|");
    const write = new RegExp(String.raw`(?<![\w.])(${names})(?:\.\w+)*(?:\.(?:set|delete|clear)\(|\s*=(?!=))`);
    for (const { text, at } of lines(file)) {
      if (state.test(text) && !SHARED.has(`${file} ${text.match(state)[1]}`))
        found.push(`${at} is module state every importer shares; make it a closure or a field of a part`);
      else if (names !== "" && write.test(text) && !SHARED.has(`${file} ${text.match(write)[1]}`))
        found.push(
          `${at} changes ${text.match(write)[1]} of vendor/, which every importer shares; change an object a part makes`,
        );
    }
  }
  none(found);
});

test("an element of index.html is looked up in ui/elements.js", () => {
  none(
    sources
      .filter((name) => !["ui/elements.js", "startup.js"].includes(name) && !name.startsWith("extensions/"))
      .flatMap((file) => lines(file).filter(({ text }) => /querySelector(All)?\(["'`]#|getElementById\(/.test(text)))
      .map(({ at }) => `${at} looks up an element by id; add it to SELECTORS in src/ui/elements.js`),
  );
});

test("an extension looks up only its own panel's elements", () => {
  const page = /\bdocument\.(?:querySelector(?:All)?|getElementById|getElementsBy\w+)\(/;
  none(
    sources
      .filter((name) => name.startsWith("extensions/"))
      .flatMap((file) => lines(file).filter(({ text }) => page.test(text)))
      .map(({ at }) => `${at} looks up an element of the page; find the extension's own in its panel, ask the viewer`),
  );
});

test("a caught error is used: reported or thrown again, never dropped", () => {
  const found = [];
  for (const file of sources) {
    const source = read(`src/${file}`);
    if (/\bcatch\s*\{/.test(source)) found.push(`src/${file} catches without naming the error`);
    for (const match of source.matchAll(/\bcatch\s*\((\w+)\)\s*\{/g)) {
      let depth = 1;
      let end = match.index + match[0].length;
      while (depth > 0) {
        if (source[end] === "{") depth += 1;
        if (source[end] === "}") depth -= 1;
        end += 1;
      }
      const body = source.slice(match.index + match[0].length, end);
      const line = source.slice(0, match.index).split("\n").length;
      if (!new RegExp(`\\b${match[1]}\\b`).test(body)) found.push(`src/${file}:${line} drops the error it catches`);
    }
  }
  none(found);
});

test("a time or a range of the core is a tunable of config.js", () => {
  const found = [];
  for (const file of sources.filter((name) => name !== "config.js" && !name.startsWith("extensions/"))) {
    for (const { text, at } of lines(file)) {
      if (/^(?:export\s+)?const\s+\w*_(?:MS|RANGE)\b/.test(text))
        found.push(`${at} is a time or a range of the core; move it to config.js`);
    }
  }
  none(found);
});

test("a file's constants come before its first function or class", () => {
  const found = [];
  for (const file of sources) {
    const all = lines(file);
    const first = all.findIndex(({ text }) =>
      /^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\b/.test(text),
    );
    for (const { text, at } of first === -1 ? [] : all.slice(first)) {
      if (/^(?:export\s+)?const\s+[A-Z][A-Z0-9_]*\s*=(?!\s*`)/.test(text))
        found.push(`${at} is a constant below ${all[first].at}; move it to the top of the file`);
    }
  }
  none(found);
});

test("no base classes, and no Manager, Service, Handler, Impl, _v2 or _fast in a name", () => {
  const found = [];
  for (const file of [...sources.map((name) => `src/${name}`), ...units, ...tools, ...checks]) {
    if (VAGUE.test(file)) found.push(`${file} has a vague name; name what the file holds`);
  }
  for (const file of sources) {
    for (const { text, at } of lines(file)) {
      if (/\bclass\s+\w+\s+extends\b/.test(text)) found.push(`${at} extends a class; hold the part and call it`);
      if (VAGUE.test(text)) found.push(`${at} names ${text.match(VAGUE)[0]}; name what it holds or does`);
    }
  }
  none(found);
});

test("a coordinate triple's name ends in _ZYX or _XYZ, in capitals", () => {
  const lower = /(?<![.\w])\w+_(?:zyx|xyz)\b/g;
  const string = /(["'`])(?:\\.|(?!\1).)*\1/g;
  none(
    [...sources.map((name) => `src/${name}`), ...units, ...tools, ...checks].flatMap((file) =>
      read(file)
        .split("\n")
        .flatMap((text, i) =>
          [...text.replace(string, "").matchAll(lower)].map(([name]) => `${file}:${i + 1} names ${name}`),
        )
        .map((found) => `${found}; end it in _ZYX, or _XYZ in the x, y, z order of three.js and GL`),
    ),
  );
});

test("an array of points names its axes' order before its shape", () => {
  const unordered = /\bpoints_(?!zyx_|xyz_)[A-Z]+3\b/g;
  none(
    [...sources.map((name) => `src/${name}`), ...units, ...tools, ...checks].flatMap((file) =>
      read(file)
        .split("\n")
        .flatMap((text, i) =>
          [...text.matchAll(unordered)].map(
            ([name]) => `${file}:${i + 1} names ${name}; name its axes, as points_zyx_${name.slice(7)}`,
          ),
        ),
    ),
  );
});

test("comments describe the present", () => {
  const found = [];
  for (const file of sources) {
    const source = read(`src/${file}`);
    for (const match of source.matchAll(/\/\/.*|\/\*[\s\S]*?\*\//g)) {
      const at = `src/${file}:${source.slice(0, match.index).split("\n").length}`;
      if (HISTORY.test(match[0])) found.push(`${at} tells history: ${match[0].match(HISTORY)[0]}`);
      if (BANDAGE.test(match[0])) found.push(`${at} describes what should be fixed: ${match[0].match(BANDAGE)[0]}`);
    }
  }
  none(found);
});

test("the page's text joins no pieces with a middle dot", () => {
  const comment = /\/\*[\s\S]*?\*\/|<!--[\s\S]*?-->/g;
  const found = [];
  for (const file of [...sources.map((name) => `src/${name}`), "index.html", "styles.css"]) {
    read(file)
      .replace(comment, (text) => text.replace(/[^\n]/g, " "))
      .split("\n")
      .forEach((text, i) => {
        if (text.replace(/(?:^|\s)\/\/.*$/, "").includes("·"))
          found.push(
            `${file}:${i + 1} joins text with a middle dot; give each piece its own element, a column or words`,
          );
      });
  }
  none(found);
});

test("a slider has a number field beside it, which takes a typed value and keeps its keys", () => {
  const found = [];
  for (const file of ["index.html", ...sources.map((name) => `src/${name}`)]) {
    const text = read(file);
    const at = (piece) => `${file}:${text.slice(0, text.indexOf(piece)).split("\n").length}`;
    const inputs = text.match(/<input\b[^>]*>/g) ?? [];
    const numbers = inputs.filter((input) => input.includes('type="number"'));
    for (const slider of inputs.filter((input) => input.includes('type="range"'))) {
      const id = /\bid="([^"]+)"/.exec(slider)?.[1];
      const camera = /\bdata-camera="([^"]+)"/.exec(slider)?.[1];
      const field = id === undefined ? `data-camera="${camera}"` : `id="${id}-number"`;
      if (!UNPAIRED_SLIDERS.includes(id) && !numbers.some((number) => number.includes(field)))
        found.push(`${at(slider)} makes a slider with no number field of ${field}; add one beside it`);
    }
    for (const [made, name] of text.matchAll(/(\w+)\.type = "range"/g))
      if (!UNPAIRED_SLIDERS.includes(`${file} ${name}`) && !new RegExp(String.raw`\b${name}Number\b`).test(text))
        found.push(`${at(made)} makes slider ${name} with no number field ${name}Number; add one beside it`);
  }
  none(found);
});

test("a file reads top to bottom: a helper is declared above the code that names it", () => {
  const names = (name) => new RegExp(String.raw`(?<![.\w$])${name}\b(?!\s*:)`);
  const found = [];
  for (const file of sources.filter((name) => name !== "main.js")) {
    const text = read(`src/${file}`).split("\n");
    const starts = text.flatMap((line, i) => (/^(?![\s})\]`/*]|$)/.test(line) ? [i] : []));
    const statements = starts.map((at, k) => ({
      at,
      code: text
        .slice(at, starts[k + 1])
        .filter((line) => !/^\s*(?:\/\/|\/\*|\*)/.test(line))
        .join("\n"),
      name: /^(?:async )?(?:function\*? ?|class )(\w+)/.exec(text[at])?.[1] ?? null,
    }));
    for (const [k, helper] of statements.entries()) {
      if (helper.name === null) continue;
      const user = statements
        .slice(0, k)
        .find(({ code, name }) => names(helper.name).test(code) && (name === null || !names(name).test(helper.code)));
      if (user !== undefined)
        found.push(
          `src/${file}:${helper.at + 1} declares ${helper.name}, named at :${user.at + 1}; move it above that`,
        );
    }
  }
  none(found);
});

test("a comment line fits the width the formatter keeps code to, since it does not wrap comments", () => {
  const width = JSON.parse(read("biome.json")).formatter.lineWidth;
  const found = [];
  for (const file of sources) {
    for (const { text, at } of lines(file)) {
      if (/^\s*(\/\/|\/\*|\*)/.test(text) && text.length > width)
        found.push(`${at} is a comment line ${text.length} wide; wrap it within ${width}`);
    }
  }
  none(found);
});

test("a file stays short enough to read in one sitting", () => {
  const found = [];
  for (const file of sources) {
    const count = read(`src/${file}`).split("\n").length - 1;
    const listed = LONG_FILES[file];
    if (listed === undefined && count > LINE_LIMIT)
      found.push(`src/${file} has ${count} lines, over ${LINE_LIMIT}; split out a part with one owner`);
    else if (count > listed)
      found.push(`src/${file} has ${count} lines, over its ${listed}; split out a part with one owner`);
    else if (count > LINE_LIMIT && count < listed)
      found.push(`src/${file} has ${count} lines; set LONG_FILES in tests/rules.test.js to ${count}`);
    else if (listed !== undefined && count <= LINE_LIMIT)
      found.push(`src/${file} has ${count} lines; drop it from LONG_FILES in tests/rules.test.js`);
  }
  none(found);
});

test("every package script names files that exist", () => {
  const found = [];
  for (const [name, script] of Object.entries(JSON.parse(read("package.json")).scripts)) {
    for (const word of script.split(/\s+/).filter((word) => /\.\w+$|\//.test(word) && !word.startsWith("-"))) {
      if (!existsSync(join(ROOT, word))) found.push(`package.json script ${name} names ${word}, which does not exist`);
    }
  }
  none(found);
});

test("a unit test runs without a page: it touches no page object", () => {
  const found = [];
  for (const file of units.filter((name) => !["tests/rules.test.js", "tests/architecture.test.js"].includes(name))) {
    read(file)
      .split("\n")
      .forEach((text, i) => {
        if (PAGE.test(text)) found.push(`${file}:${i + 1} reaches ${text.match(PAGE)[0]}; test a pure function`);
      });
  }
  none(found);
});

test("a tool takes argparse and a check is one check(page, out)", () => {
  const imported = new Set(
    [...tools, ...checks].flatMap((file) =>
      [...read(file).matchAll(/^from (\w+) import/gm)].map((m) => `tools/${m[1]}.py`),
    ),
  );
  const found = [];
  for (const file of [...tools, ...checks]) {
    const text = read(file).replace(/^# \/\/\/ script\n(?:#.*\n)*?# \/\/\/\n/, "");
    if (file.startsWith("tools/") && !imported.has(file) && !text.includes("argparse.ArgumentParser("))
      found.push(`${file} takes no argparse`);
    if (file.startsWith("tests/") && !/^async def check\(page, out\):$/m.test(text))
      found.push(`${file} holds no async def check(page, out)`);
  }
  none(found);
});

test("a Python except uses the error it catches, or names one that is retried", () => {
  const found = [];
  for (const file of [...tools, ...checks]) {
    const all = read(file).split("\n");
    all.forEach((text, i) => {
      const clause = text.match(/^(\s*)except\b([^:]*):/);
      if (clause === null) return;
      const named = clause[2].match(/\bas (\w+)$/);
      if (named === null) {
        if (!RETRIED.includes(clause[2].trim()))
          found.push(`${file}:${i + 1} drops the error it catches: ${text.trim()}`);
        return;
      }
      const body = [];
      for (
        let j = i + 1;
        j < all.length && (all[j].trim() === "" || all[j].match(/^\s*/)[0].length > clause[1].length);
        j++
      ) {
        body.push(all[j]);
      }
      if (!body.some((line) => new RegExp(`\\b${named[1]}\\b`).test(line)))
        found.push(`${file}:${i + 1} drops the error it catches: ${text.trim()}`);
    });
  }
  none(found);
});

test("a browser check drives the real page: it replaces no browser API", () => {
  const found = [];
  for (const file of [...checks, "tools/page.py"]) {
    read(file)
      .split("\n")
      .forEach((text, i) => {
        if (REPLACED.test(text)) found.push(`${file}:${i + 1} replaces a browser API: ${text.trim()}`);
      });
  }
  none(found);
});

test("a browser check drives one extension's elements, its own: tests/<name>.e2e.py or tests/<name>-*.e2e.py", () => {
  const id = new RegExp(String.raw`(?:#|type_into\(\s*["'])(${extensions.join("|")})-`, "g");
  const found = [];
  for (const file of checks) {
    read(file)
      .split("\n")
      .forEach((text, i) => {
        for (const [, name] of text.matchAll(id))
          if (!owns(name, file))
            found.push(
              `${file}:${i + 1} drives the #${name}- elements of extension ${name}; move it to tests/${name}.e2e.py, ` +
                "or draw through page.extension_viewer() as an extension would",
            );
      });
  }
  none(found);
});

test("every extension has a browser check of its own, which a commit of its files runs", () => {
  none(
    extensions
      .filter((name) => !checks.some((check) => owns(name, check)))
      .map((name) => `src/extensions/${name}.js has no browser check; write tests/${name}.e2e.py`),
  );
});

test("a browser check reads a tunable of config.js with page.config, never a copy of it", () => {
  const tunables = new Set([...read("src/config.js").matchAll(/^export const (\w+)/gm)].map((m) => m[1]));
  const found = [];
  for (const file of [...checks, ...tools.filter((name) => name !== "tools/page.py")]) {
    read(file)
      .split("\n")
      .forEach((text, i) => {
        const number = text.match(/^\s*(\w+)\s*=\s*[-\d.]+\b[^#]*(?:#\s*(.*))?$/);
        if (number !== null && (tunables.has(number[1]) || number[2]?.startsWith("config.js")))
          found.push(`${file}:${i + 1} copies a value of src/config.js; read it with await page.config(name)`);
        if (/(["'`])[^"'`]*config\.js\1/.test(text))
          found.push(`${file}:${i + 1} imports src/config.js in the page; read a value with await page.config(name)`);
      });
  }
  none(found);
});
