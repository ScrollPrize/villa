// A level's scale_ZYX and translation_ZYX are its NGFF transform, 1 and 0 on an axis the array lacks. Its
// downsample_ZYX and offset_ZYX are the same in voxels of the level that is finest on every spatial axis.

import { VIEWER_CHANNEL_LIMIT, ZARR_SCALE_LIMIT } from "../config.js";
import { validateZarray } from "./zarray.js";

const SUPPORTED_AXES = new Set(["t", "c", "z", "y", "x"]);
const AXIS_UNIT_LENGTH_LIMIT = 64;

export function declaredDatasetPaths(attrs) {
  if (!Array.isArray(attrs.multiscales) || attrs.multiscales.length === 0) {
    throw new Error("A viewer source must declare a multiscales entry.");
  }
  const datasets = attrs.multiscales[0]?.datasets;
  if (!Array.isArray(datasets) || datasets.length === 0) {
    throw new Error("Multiscale metadata must declare at least one dataset.");
  }
  if (datasets.length > ZARR_SCALE_LIMIT) {
    throw new Error(`Multiscale metadata exceeds the viewer limit of ${ZARR_SCALE_LIMIT} datasets.`);
  }
  const paths = datasets.map((dataset) => {
    if (
      dataset === null ||
      typeof dataset !== "object" ||
      Array.isArray(dataset) ||
      typeof dataset.path !== "string" ||
      dataset.path.length === 0 ||
      dataset.path.startsWith("/") ||
      dataset.path.split("/").some((part) => part === "" || part === "." || part === "..")
    ) {
      throw new Error("Every multiscale dataset needs a nonempty relative path without empty or parent components.");
    }
    return dataset.path;
  });
  if (new Set(paths).size !== paths.length) throw new Error("Multiscale dataset paths must be unique.");
  return paths;
}

function declaredAxes(rawAxes) {
  if (
    !Array.isArray(rawAxes) ||
    rawAxes.length === 0 ||
    rawAxes.some((axis) => axis === null || typeof axis !== "object" || Array.isArray(axis))
  ) {
    throw new Error("Multiscale metadata must declare nonempty NGFF axis objects.");
  }
  const axes = rawAxes.map((axis) => (typeof axis.name === "string" ? axis.name.toLowerCase() : ""));
  if (new Set(axes).size !== axes.length || axes.some((axis) => !SUPPORTED_AXES.has(axis))) {
    throw new Error(`Unsupported, absent, or repeated axes: ${axes.join(",")}.`);
  }
  if (!axes.includes("y") || !axes.includes("x")) {
    throw new Error(`Axes must include y and x: ${axes.join(",")}.`);
  }
  const nonspatial = ["t", "c"].filter((axis) => axes.includes(axis));
  if (
    axes.slice(0, nonspatial.length).some((axis, index) => axis !== nonspatial[index]) ||
    axes.slice(nonspatial.length).some((axis) => !"zyx".includes(axis))
  ) {
    throw new Error("NGFF axes must order time, channel, then spatial dimensions.");
  }
  const units = {};
  for (let index = 0; index < axes.length; index++) {
    const name = axes[index];
    const raw = rawAxes[index];
    if (raw.unit !== undefined && raw.unit !== null) {
      if (
        typeof raw.unit !== "string" ||
        raw.unit.length === 0 ||
        raw.unit.length > AXIS_UNIT_LENGTH_LIMIT ||
        /[\u0000-\u001f\u007f]/u.test(raw.unit)
      ) {
        throw new Error(`Axis ${name} has an invalid unit.`);
      }
    }
    units[name] = raw.unit ?? null;
  }
  return { axes, names: rawAxes.map((axis) => axis.name), units };
}

function declaredTransform(transforms, axes, label) {
  if (!Array.isArray(transforms) || transforms.length < 1 || transforms.length > 2) {
    throw new Error(`${label} must declare exactly one scale followed by optional translation.`);
  }
  if (transforms.some((transform) => transform === null || typeof transform !== "object" || Array.isArray(transform))) {
    throw new Error(`${label} contains a malformed coordinate transform.`);
  }
  if (transforms[0].type !== "scale" || (transforms.length === 2 && transforms[1].type !== "translation")) {
    throw new Error(`${label} supports exactly one scale followed by optional translation.`);
  }
  const expectedKeys =
    transforms.length === 1
      ? [["scale", "type"]]
      : [
          ["scale", "type"],
          ["translation", "type"],
        ];
  if (
    transforms.some((transform, index) => Object.keys(transform).sort().join("\0") !== expectedKeys[index].join("\0"))
  ) {
    throw new Error(`${label} coordinate transformation fields are unsupported.`);
  }
  if (!Array.isArray(transforms[0].scale) || transforms[0].scale.length !== axes.length) {
    throw new Error(`${label} scale must match axes ${axes.join(",")}.`);
  }
  const scale = [...transforms[0].scale];
  if (scale.some((value) => typeof value !== "number" || !Number.isFinite(value) || value <= 0)) {
    throw new Error(`${label} scale values must be finite and positive.`);
  }
  const translation =
    transforms.length === 1
      ? Array(axes.length).fill(0)
      : Array.isArray(transforms[1].translation) && transforms[1].translation.length === axes.length
        ? [...transforms[1].translation]
        : null;
  if (translation === null || translation.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error(`${label} translation must contain one finite value per axis.`);
  }
  const channelAxis = axes.indexOf("c");
  if (channelAxis !== -1 && (scale[channelAxis] !== 1 || translation[channelAxis] !== 0)) {
    throw new Error(`${label} c transform must remain scale 1 and translation 0.`);
  }
  return { scale, translation };
}

function sourcePhysicalMetadata(axisMetadata, finestSpatial, finestTranslation) {
  return {
    scale_ZYX: ["z", "y", "x"].map((axis) => {
      const index = axisMetadata.axes.indexOf(axis);
      return index === -1 || axisMetadata.units[axis] === null ? null : finestSpatial[axis];
    }),
    translation_ZYX: ["z", "y", "x"].map((axis) => {
      const index = axisMetadata.axes.indexOf(axis);
      return index === -1 || axisMetadata.units[axis] === null ? null : finestTranslation[index];
    }),
    units_ZYX: ["z", "y", "x"].map((axis) => axisMetadata.units[axis] ?? null),
  };
}

function axisValue(values, axes, axis, missing) {
  const index = axes.indexOf(axis);
  return index === -1 ? missing : values[index];
}

function declaredLevels(multiscale, paths, axisMetadata, zarrays) {
  const axes = axisMetadata.axes;
  const sharedTransform =
    "coordinateTransformations" in multiscale
      ? declaredTransform(multiscale.coordinateTransformations, axes, "Multiscale")
      : { scale: Array(axes.length).fill(1), translation: Array(axes.length).fill(0) };
  const rawScales = multiscale.datasets.map((dataset, index) => {
    const path = paths[index];
    const zarray = zarrays[path];
    const validatedArray = validateZarray(zarray, axisMetadata.names, path);
    const datasetTransform = declaredTransform(dataset.coordinateTransformations, axes, `Dataset ${path}`);
    const scale = datasetTransform.scale.map((value, axis) => value * sharedTransform.scale[axis]);
    const translation = datasetTransform.translation.map(
      (value, axis) => value * sharedTransform.scale[axis] + sharedTransform.translation[axis],
    );
    return {
      path,
      axes,
      storageShape: [...zarray.shape],
      storageChunks: validatedArray.chunks,
      chunkSeparator: validatedArray.chunkSeparator,
      storageFormat: zarray.zarr_format,
      compressor: validatedArray.compressor,
      fillValue: validatedArray.fillValue,
      transformScale: scale,
      transformTranslation: translation,
      numericType: validatedArray.numericType,
    };
  });
  if (new Set(rawScales.map((scale) => scale.numericType.dtype)).size !== 1) {
    throw new Error("Every scale must expose the same numeric dtype.");
  }
  const finestSpatial = Object.fromEntries(
    ["z", "y", "x"].map((axis) => {
      const axisIndex = axes.indexOf(axis);
      return [axis, axisIndex === -1 ? 1 : Math.min(...rawScales.map((scale) => scale.transformScale[axisIndex]))];
    }),
  );
  const finest = rawScales.find((scale) =>
    ["z", "y", "x"].every((axis) => {
      const index = axes.indexOf(axis);
      return index === -1 || scale.transformScale[index] === finestSpatial[axis];
    }),
  );
  if (!finest) throw new Error("No dataset is finest on every spatial axis.");
  const finestTranslation_ZYX = ["z", "y", "x"].map((axis) => axisValue(finest.transformTranslation, axes, axis, 0));
  const scales = rawScales.map((scale) => {
    const scale_ZYX = ["z", "y", "x"].map((axis) => axisValue(scale.transformScale, axes, axis, 1));
    const translation_ZYX = ["z", "y", "x"].map((axis) => axisValue(scale.transformTranslation, axes, axis, 0));
    return {
      ...scale,
      shape_ZYX: ["z", "y", "x"].map((axis) => axisValue(scale.storageShape, axes, axis, 1)),
      chunks_ZYX: ["z", "y", "x"].map((axis) => axisValue(scale.storageChunks, axes, axis, 1)),
      scale_ZYX,
      translation_ZYX,
      downsample_ZYX: scale_ZYX.map((value, axis) => value / finestSpatial["zyx"[axis]]),
      offset_ZYX: translation_ZYX.map(
        (value, axis) => (value - finestTranslation_ZYX[axis]) / finestSpatial["zyx"[axis]],
      ),
    };
  });
  return { scales, finestSpatial, finestTranslation: finest.transformTranslation };
}

export function readVolumeDocument(attrs, zarrays) {
  const paths = declaredDatasetPaths(attrs);
  const multiscale = attrs.multiscales[0];
  const axisMetadata = declaredAxes(multiscale.axes);
  const axes = axisMetadata.axes;
  const { scales, finestSpatial, finestTranslation } = declaredLevels(multiscale, paths, axisMetadata, zarrays);
  const channelCount = axes.includes("c") ? scales[0].storageShape[axes.indexOf("c")] : 1;
  if (channelCount > VIEWER_CHANNEL_LIMIT) {
    throw new Error(`Source has ${channelCount} channels, exceeding the viewer limit of ${VIEWER_CHANNEL_LIMIT}.`);
  }
  if (scales.some((scale) => (axes.includes("c") ? scale.storageShape[axes.indexOf("c")] : 1) !== channelCount)) {
    throw new Error("Every scale must expose the same channel count.");
  }
  const timeAxis = axes.indexOf("t");
  const timeCount = timeAxis === -1 ? 1 : scales[0].storageShape[timeAxis];
  if (scales.some((scale) => (timeAxis === -1 ? 1 : scale.storageShape[timeAxis]) !== timeCount)) {
    throw new Error("Every scale must expose the same time-point count.");
  }
  const transportDtype = scales[0].numericType.dtype;
  const labels = attrs.omero?.channels?.map((channel) => channel.label);
  const named = labels?.length === channelCount && labels.every((label) => typeof label === "string");
  const physical = sourcePhysicalMetadata(axisMetadata, finestSpatial, finestTranslation);
  const metadata = {
    axes,
    sourceDtype: transportDtype,
    channelCount,
    channelNames: named ? labels : null,
    timeCount,
    displayEncoding: transportDtype === "|u1" ? { kind: "identity_uint8" } : null,
    physicalScale_ZYX: physical.scale_ZYX,
    physicalTranslation_ZYX: physical.translation_ZYX,
    physicalUnits_ZYX: physical.units_ZYX,
  };
  return { scales, metadata };
}
