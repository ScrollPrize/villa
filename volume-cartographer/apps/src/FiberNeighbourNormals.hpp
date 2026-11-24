#pragma once
#include "vc/fiber_tracer/FiberNeighbourNormals.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <vector>

using FiberStripSample = vc::fiber_tracer::FiberStripSample;

// Strip samples (vc/fiber_tracer/FiberNeighbourNormals.hpp) addressed by fiber uuid strings.
void writeFiberStripSamples(const std::filesystem::path& output, const std::vector<std::string>& ids,
    const std::vector<std::vector<FiberStripSample>>& samples);

// Samples per fiber of `ids` (empty for fibers the file does not list).
std::vector<std::vector<FiberStripSample>> readFiberStripSamples(const std::filesystem::path& path,
    const std::vector<std::string>& ids, size_t& unknownFibers);

// Normal and sign at every strip sample of the query fibers, inferred only from
// the signed strip samples of OTHER fibers within a radius
// (vc::fiber_tracer::FiberNeighbourField). Config keys: strip_samples, fiber_signs (JSON uuid ->
// sign), neighbour_queries (JSON uuid -> uuids to exclude), neighbour_variants
// ([{falloff, sigma or scales (tangent, side, normal), parallel_scales, frame (isotropic, own,
// prediction, two_pass, fiber), radius, first_sigma, fiber_cap}]), neighbour_normals_output,
// workers; optionally neighbour_samples_output (+ neighbour_samples_variant):
// the signed inferred normals of one variant as strip samples. Returns the run summary.
nlohmann::json inferFiberNeighbourNormals(const nlohmann::json& cfg);

// Signed strip samples for consumers outside the pipeline (the scaffold solve's
// neighbour normals): the samples of `strip_samples` with every normal
// multiplied by its fiber's sign of the `sign_store`'s normal-sign run (`sign_run`
// selects one of several), unsigned fibers left out. Writes
// `signed_strip_samples_output` and, next to it, `<output>.json` stating what it
// is: `coordinate_space` and `volume_uuid` (the frame of its points, as the
// consuming store names the volume), the sign store and run, the counts. Returns that statement.
nlohmann::json writeSignedFiberStripSamples(const nlohmann::json& cfg);

// Crossings of every fiber with every other fiber's normal ribbon
// (vc/fiber_tracer/FiberRibbonCrossings.hpp) over the signed strip samples of
// `strip_samples`. Keys: ribbon_crossings_output, ribbon_half_height (128),
// ribbon_max_tangent_cosine (0.7), workers. Writes 11 doubles per crossing
// (ribbon fiber index in file order, crossing fiber index, ribbon arc, fiber
// arc, height, direction, tangent cosine, normal dot, point XYZ) and
// `<output>.json`; the crossings are ordered by (ribbon, fiber, fiber arc). Their
// encounters (ribbon_encounter_arc_gap, 3000) go to `<output>.encounters`, 7 doubles each:
// ribbon, fiber, first crossing, count, closest crossing, crossings left, remaining crossing (-1).
// Returns the run summary.
nlohmann::json writeFiberRibbonCrossings(const nlohmann::json& cfg);

// The multi-crossing findings the sign stage filters by (vc/fiber_tracer/FiberRibbonCrossings.hpp):
// every perpendicular event of the constraint store `database` with a valid normal on both sides is
// asked on the normal surfaces of its two fibers, built from `strip_samples` (the extraction's
// `strip_samples_output`; the normals need no sign). Writes `crossing_filter_output`: one 24-byte
// record per such event, ascending by id - int64 event id, int64 finding bits (1 a surface crossing
// matches, 2 several crossings, 4 nothing left, 8 mixed, 16 mixed within crossing_filter_near_height,
// 32 several left; a finding holds when either surface states it) and the |normal dot| of the nearest
// strong crossing that states the other relation than the event (0: none) - and `<output>.json` with
// the store, the parameters and the counts. Which findings drop an event is the sign stage's choice.
// Keys: ribbon_half_height (512), ribbon_encounter_arc_gap (3000), crossing_filter_match_tolerance
// (64), crossing_filter_minimum_dot (0.316), crossing_filter_near_height (128), workers.
nlohmann::json writeFiberCrossingFilter(const nlohmann::json& cfg);
