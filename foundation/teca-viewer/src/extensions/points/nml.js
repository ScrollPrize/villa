// A node's x, y and z are voxels of the finest level, whole numbers at voxel centres. teca.strokeId names its stroke.
// teca.pane and teca.normal_ZYX, the normal in the finest level's axes, give the view it was placed in.

import { unzipped, zipped } from "../../zip.js";
import { apply, emptyPatch, emptySkeleton } from "./skeleton.js";
import { gapOf, placeOf, treeName, treeRole } from "./targets.js";

const TAG = /<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<(\/?)([\w.:-]+)((?:\s+[\w.:-]+\s*=\s*(?:"[^"]*"|'[^']*'))*)\s*(\/?)>/g;
const ATTRIBUTE = /([\w.:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const NODE_AT = /(<node [^>]*?) x="([^"]*)" y="([^"]*)" z="([^"]*)"/g;
// Three decimals hold a normal within 0.05°.
const PANES = new Set(["xy", "zy", "xz", "surface"]);
const NORMAL_DECIMALS = 3;
const COARSER = /-level-([1-9]\d*)\.nml$/i;
const UNSAFE = /[<>:"/\\|?*\u0000-\u001f]/g;
const DOWNLOAD_NAME_LENGTH = 180;
const NAMES_LISTED = 4;
const LINE_SHOWN = 120;

function decoded(value) {
  return value.replace(/&(?:#x([0-9a-f]+)|#([0-9]+)|(\w+));/gi, (entity, hex, decimal, name) => {
    if (hex !== undefined) return String.fromCodePoint(Number.parseInt(hex, 16));
    if (decimal !== undefined) return String.fromCodePoint(Number(decimal));
    if (!(name in ENTITIES)) throw new Error(`NML has an unknown entity ${entity}.`);
    return ENTITIES[name];
  });
}

function escaped(value) {
  return value.replace(/[&<>"]/g, (character) => ESCAPES[character]);
}

function required(attributes, key, tag) {
  const value = attributes.get(key);
  if (value === undefined || value.trim() === "") throw new Error(`An NML <${tag}> has no ${key}.`);
  return value;
}

function number(attributes, key, tag) {
  const value = Number(required(attributes, key, tag));
  if (!Number.isFinite(value)) throw new Error(`An NML <${tag}> has ${key} ${attributes.get(key)}, not a number.`);
  return value;
}

function id(attributes, key, tag) {
  const value = number(attributes, key, tag);
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`An NML <${tag}> has ${key} ${value}, not an id.`);
  return value;
}

function voxelOf(level, point_ZYX) {
  return point_ZYX.map(
    (world, axis) =>
      (world - level.offset_ZYX[axis] - (level.downsample_ZYX[axis] - 1) / 2) / level.downsample_ZYX[axis],
  );
}

function roleName(place, role, fiber) {
  const number = (letter, value) => `${letter}${value === null ? "N" : value}`;
  if (role === "air") {
    const above = place.winding === null ? null : place.winding + 1;
    return [number("W", place.winding), number("W", above), "AIR"].join("_");
  }
  const side = place.side === null ? [] : [place.side.toUpperCase()];
  const family = place.side === null ? [] : [place.side === "recto" ? "HZ" : "VT"];
  const kind =
    role === "fiber"
      ? [...family, ...side, "FIBER", String(fiber)]
      : [...side, "SURFACE", role === "ink" ? "INK" : "NOINK"];
  return [number("S", place.sheet), number("W", place.winding), ...kind].join("_");
}

export function fileName(groups, tree) {
  const role = treeRole(groups, tree);
  if (role === null) return tree.name;
  return roleName(placeOf(groups, tree.group) ?? gapOf(groups, tree.group), role.role, role.fiber);
}

function readName(groups, group, name) {
  const gap = gapOf(groups, group);
  if (gap !== null) return roleName(gap, "air") === name ? treeName("air") : name;
  const place = placeOf(groups, group);
  if (place === null) return name;
  const fiber = /_FIBER_([1-9]\d*)$/.exec(name);
  const role = fiber !== null ? { role: "fiber", fiber: Number(fiber[1]) } : null;
  for (const { role: kind, fiber: number } of [role, { role: "ink" }, { role: "no ink" }].filter(Boolean))
    if (roleName(place, kind, number) === name) return treeName(kind, number);
  return name;
}

export function inside(level, point_ZYX) {
  return voxelOf(level, point_ZYX).every((voxel, axis) => voxel >= 0 && voxel <= level.shape_ZYX[axis] - 1);
}

export function readNml(text, level) {
  const patch = emptyPatch();
  const groups = new Map();
  const within = [];
  const trees = new Map();
  const nodes = new Map();
  const edges = [];
  const branchpoints = [];
  const comments = [];
  let active = null;
  let root = null;
  const open = [];
  let tree = null;
  let end = 0;
  for (const match of text.matchAll(TAG)) {
    const between = text.slice(end, match.index).trim();
    if (between !== "") throw new Error(`NML has ${JSON.stringify(between.slice(0, 40))} outside a tag.`);
    end = match.index + match[0].length;
    const [, closing, tag, attributeText, empty] = match;
    if (tag === undefined) continue;
    if (closing === "/") {
      const innermost = open.pop();
      if (innermost !== tag)
        throw new Error(
          `NML has </${tag}> where ${innermost === undefined ? "no element" : `<${innermost}>`} is open.`,
        );
      if (tag === "thing") tree = null;
      if (tag === "group") within.pop();
      continue;
    }
    if (open.length === 0 && root !== null) throw new Error(`NML has a second root, <${tag}>.`);
    if (open.length === 0) root = tag;
    const parentTag = open.at(-1);
    if (empty === "") open.push(tag);
    const attributes = new Map(
      [...attributeText.matchAll(ATTRIBUTE)].map(([, key, double, single]) => [key, decoded(double ?? single)]),
    );
    if (tag === "group") {
      if (parentTag !== "groups" && parentTag !== "group") throw new Error("An NML <group> is outside <groups>.");
      const group = id(attributes, "id", tag);
      if (groups.has(group)) throw new Error(`NML has group ${group} twice.`);
      groups.set(group, { id: group, name: attributes.get("name") ?? "", parent: within.at(-1) ?? null });
      if (empty === "") within.push(group);
    } else if (tag === "thing") {
      tree = id(attributes, "id", tag);
      if (trees.has(tree)) throw new Error(`NML has tree ${tree} twice.`);
      const color_RGB = attributes.has("color.r")
        ? ["r", "g", "b"].map((key) => number(attributes, `color.${key}`, tag))
        : null;
      const group = attributes.has("groupId") ? id(attributes, "groupId", tag) : null;
      trees.set(tree, { id: tree, name: attributes.get("name") ?? `Class ${tree}`, color_RGB, group });
    } else if (tag === "node" || tag === "edge") {
      if (tree === null) throw new Error(`An NML <${tag}> is outside a <thing>.`);
      if (tag === "edge") edges.push([id(attributes, "source", tag), id(attributes, "target", tag)]);
      else {
        const node = id(attributes, "id", tag);
        if (nodes.has(node)) throw new Error(`NML has node ${node} twice.`);
        const voxel_ZYX = ["z", "y", "x"].map((axis) => number(attributes, axis, tag));
        const point_ZYX = voxel_ZYX.map(
          (voxel, axis) =>
            level.offset_ZYX[axis] + voxel * level.downsample_ZYX[axis] + (level.downsample_ZYX[axis] - 1) / 2,
        );
        if (!inside(level, point_ZYX))
          throw new Error(`NML node ${node} at x, y, z ${voxel_ZYX.toReversed().join(", ")} is outside this CT.`);
        const radius = attributes.has("radius") ? number(attributes, "radius", tag) : 1;
        const stroke = attributes.get("teca.strokeId") ?? null;
        const pane = attributes.get("teca.pane") ?? null;
        const normal = attributes.get("teca.normal_ZYX") ?? null;
        if ((pane === null) !== (normal === null))
          throw new Error(`NML node ${node} has ${pane === null ? "teca.normal_ZYX" : "teca.pane"} alone.`);
        let view = null;
        if (pane !== null) {
          const normal_ZYX = normal.split(",").map(Number);
          if (!PANES.has(pane)) throw new Error(`NML node ${node} has teca.pane ${pane}, not xy, zy, xz or surface.`);
          if (normal_ZYX.length !== 3 || !normal_ZYX.every(Number.isFinite))
            throw new Error(`NML node ${node} has teca.normal_ZYX ${normal}, not three numbers.`);
          view = { pane, normal_ZYX };
        }
        nodes.set(node, { id: node, tree, point_ZYX, radius, stroke, comment: null, view });
      }
    } else if (tag === "branchpoint") branchpoints.push(id(attributes, "id", tag));
    else if (tag === "comment") comments.push([id(attributes, "node", tag), required(attributes, "content", tag)]);
    else if (tag === "activeNode") active = id(attributes, "id", tag);
  }
  if (text.slice(end).trim() !== "")
    throw new Error(`NML has ${JSON.stringify(text.slice(end).trim().slice(0, 40))} outside a tag.`);
  if (open.length > 0) throw new Error(`NML leaves <${open.at(-1)}> open.`);
  if (root !== "things") throw new Error("The file has no NML <things>.");
  for (const record of trees.values()) {
    if (record.group !== null && !groups.has(record.group))
      throw new Error(`NML tree ${record.id} is in group ${record.group}, which the file does not have.`);
    record.name = readName(groups, record.group, record.name);
  }
  for (const [node, content] of comments) {
    if (!nodes.has(node)) throw new Error(`An NML comment is on node ${node}, which the file does not have.`);
    nodes.set(node, { ...nodes.get(node), comment: content });
  }
  for (const [source, target] of edges) {
    if (!nodes.has(source) || !nodes.has(target))
      throw new Error(`NML edge ${source}–${target} joins a node the file does not have.`);
    if (source === target || nodes.get(source).tree !== nodes.get(target).tree)
      throw new Error(`NML edge ${source}–${target} does not join two nodes of one tree.`);
    patch.edges.push([[source, target], false, true]);
  }
  for (const node of branchpoints)
    if (!nodes.has(node)) throw new Error(`NML branchpoint ${node} is a node the file does not have.`);
  if (active !== null && !nodes.has(active))
    throw new Error(`NML active node ${active} is a node the file does not have.`);
  patch.groups = [...groups].map(([key, record]) => [key, null, record]);
  patch.trees = [...trees].map(([key, record]) => [key, null, record]);
  patch.nodes = [...nodes].map(([key, record]) => [key, null, record]);
  patch.branchpoints = [[], branchpoints];
  patch.selected = [null, active];
  const skeleton = emptySkeleton();
  apply(skeleton, patch);
  return skeleton;
}

export function writeNml(skeleton, name, level) {
  const lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<things>", "  <parameters>"];
  lines.push(`    <experiment name="${escaped(name)}"/>`);
  if (skeleton.selected !== null) lines.push(`    <activeNode id="${skeleton.selected}"/>`);
  lines.push("  </parameters>");
  const members = new Map([...skeleton.trees.keys()].map((tree) => [tree, []]));
  for (const node of skeleton.nodes.values()) members.get(node.tree).push(node);
  for (const tree of skeleton.trees.values()) {
    const color =
      tree.color_RGB === null
        ? ""
        : `${tree.color_RGB.map((value, index) => ` color.${"rgb"[index]}="${value}"`).join("")} color.a="1"`;
    const group = tree.group === null ? "" : ` groupId="${tree.group}"`;
    const name = escaped(fileName(skeleton.groups, tree));
    lines.push(`  <thing id="${tree.id}" name="${name}"${color}${group}>`, "    <nodes>");
    for (const node of members.get(tree.id)) {
      const voxel_ZYX = voxelOf(level, node.point_ZYX);
      const stroke = node.stroke === null ? "" : ` teca.strokeId="${escaped(node.stroke)}"`;
      const normal = node.view?.normal_ZYX.map((value) => Number(value.toFixed(NORMAL_DECIMALS))).join(",");
      const view = node.view === null ? "" : ` teca.pane="${node.view.pane}" teca.normal_ZYX="${normal}"`;
      lines.push(
        `      <node id="${node.id}" radius="${node.radius}" x="${voxel_ZYX[2]}" y="${voxel_ZYX[1]}" z="${voxel_ZYX[0]}"${stroke}${view}/>`,
      );
    }
    lines.push("    </nodes>", "    <edges>");
    for (const node of members.get(tree.id))
      for (const other of skeleton.links.get(node.id) ?? [])
        if (node.id < other) lines.push(`      <edge source="${node.id}" target="${other}"/>`);
    lines.push("    </edges>", "  </thing>");
  }
  lines.push("  <branchpoints>");
  for (const node of skeleton.branchpoints) lines.push(`    <branchpoint id="${node}"/>`);
  lines.push("  </branchpoints>", "  <comments>");
  for (const node of skeleton.nodes.values())
    if (node.comment !== null) lines.push(`    <comment node="${node.id}" content="${escaped(node.comment)}"/>`);
  lines.push("  </comments>", "  <groups>");
  const children = new Map([[null, []], ...[...skeleton.groups.keys()].map((group) => [group, []])]);
  for (const group of skeleton.groups.values()) children.get(group.parent).push(group);
  const nest = (parent, indent) => {
    for (const group of children.get(parent)) {
      const start = `${indent}<group id="${group.id}" name="${escaped(group.name)}"`;
      if (children.get(group.id).length === 0) lines.push(`${start}/>`);
      else {
        lines.push(`${start}>`);
        nest(group.id, `${indent}  `);
        lines.push(`${indent}</group>`);
      }
    }
  };
  nest(null, "    ");
  lines.push("  </groups>", "</things>", "");
  return lines.join("\n");
}

function float32Text(value) {
  const rounded = Math.fround(value);
  return String(Object.is(rounded, -0) ? 0 : rounded);
}

function scaledNml(text, scales, index, units_ZYX) {
  const { downsample_ZYX, offset_ZYX } = scales[index];
  const centre_ZYX = offset_ZYX.map((offset, axis) => offset + (downsample_ZYX[axis] - 1) / 2);
  const scaled = (voxel, axis) => float32Text((Math.fround(Number(voxel)) - centre_ZYX[axis]) / downsample_ZYX[axis]);
  const oneUnit = units_ZYX.every((unit) => typeof unit === "string" && unit !== "") && new Set(units_ZYX).size === 1;
  const size_ZYX = scales[0].scale_ZYX.map((size, axis) => size * downsample_ZYX[axis]);
  const scale = oneUnit
    ? `    <scale x="${size_ZYX[2]}" y="${size_ZYX[1]}" z="${size_ZYX[0]}" unit="${escaped(units_ZYX[0])}"/>\n`
    : "";
  return text
    .replace("  <parameters>\n", `  <parameters>\n${scale}`)
    .replace(NODE_AT, (_, start, x, y, z) => `${start} x="${scaled(x, 2)}" y="${scaled(y, 1)}" z="${scaled(z, 0)}"`);
}

function scaledName(name, index) {
  const safe = `${name}-edited.nml`.replace(UNSAFE, "_");
  const kept = safe.length <= DOWNLOAD_NAME_LENGTH ? safe : `${safe.slice(0, DOWNLOAD_NAME_LENGTH - 4)}.nml`;
  return kept.replace(/\.nml$/i, `-level-${index}.nml`);
}

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

export function savedZip(skeleton, name, scales, units_ZYX) {
  const encoder = new TextEncoder();
  function* files() {
    const finest = writeNml(skeleton, name, scales[0]);
    yield { name: `${name}.nml`, bytes: encoder.encode(finest) };
    for (let index = 1; index < scales.length; index++)
      yield { name: scaledName(name, index), bytes: encoder.encode(scaledNml(finest, scales, index, units_ZYX)) };
  }
  return zipped(files(), nextTask, true);
}

export async function readSaved(bytes, name, scales, units_ZYX) {
  if (bytes[0] !== 0x50 || bytes[1] !== 0x4b) {
    if (COARSER.test(name))
      throw new Error(
        `${name} is in a coarser level's voxels, which Points would read as the finest level's; open the zip Save ` +
          "wrote, or its NML not named -level-<n>.",
      );
    return readNml(new TextDecoder().decode(bytes), scales[0]);
  }
  const files = await unzipped(bytes);
  const finest = files.filter(({ name }) => name.toLowerCase().endsWith(".nml") && !COARSER.test(name));
  if (finest.length !== 1 || files.some(({ name }) => !name.toLowerCase().endsWith(".nml"))) {
    const names = files.map(({ name }) => name);
    const listed =
      names.length > NAMES_LISTED + 1
        ? [...names.slice(0, NAMES_LISTED), `${names.length - NAMES_LISTED} more`]
        : names;
    throw new Error(
      `The zip holds ${listed.join(", ") || "no file"}, where Points reads one NML not named -level-<n>, with ` +
        "nothing beside it but -level-<n>.nml files.",
    );
  }
  const decoder = new TextDecoder();
  const text = decoder.decode(finest[0].bytes);
  for (const file of files) {
    const match = COARSER.exec(file.name);
    if (match === null) continue;
    const index = Number(match[1]);
    if (index >= scales.length)
      throw new Error(`The zip's ${file.name} is of level ${index}, and this CT's coarsest is ${scales.length - 1}.`);
    await nextTask();
    const made = scaledNml(text, scales, index, units_ZYX);
    const held = decoder.decode(file.bytes);
    if (held === made) continue;
    const [madeLines, heldLines] = [made.split("\n"), held.split("\n")];
    let line = 0;
    while (heldLines[line] === madeLines[line]) line++;
    const shown = (content) => (content === undefined ? "the file's end" : content.trim().slice(0, LINE_SHOWN));
    throw new Error(
      `The zip's ${file.name} is not the level ${index} Save makes of ${finest[0].name}: its line ${line + 1} is ` +
        `${shown(heldLines[line])}, where Save writes ${shown(madeLines[line])}. Points reads ${finest[0].name} alone ` +
        "and would drop the change.",
    );
  }
  return readNml(text, scales[0]);
}
