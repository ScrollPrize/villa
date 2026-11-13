#pragma once
// The reconstruction of an extended-field-of-view helical scan pair from its raw folders (helirec-reconstruct;
// Python helirec.recon.HelicalRecon binds it): the scans (calibration/scan.hpp), the geometry calibration (given,
// cached or made: calibration/geometry_calibration.hpp), the intensity calibration in the frame it merges the scans
// in (calibration/intensity.hpp), every projection's terms (geometry.hpp), and a SliceReconstructor of the
// parameters (pipeline.hpp), with the slices' addressing and the provenance of everything that defines the result.
//
// Parameters (--set KEY=VALUE; a JSON object, keys in a fixed order):
//   mode                 "corrected" (default): attenuation in cm^-1, periodic opposite-view rebalance, ramp padding
//                        reflected about the true axis, mending over the merged columns the scans cover, the fixed
//                        unit-DC Paganin kernel; no projection filters by default.
//                        "legacy_reference": the reference workflow's conventions (Night Rail BM18 / nabu-helical /
//                        nabuxx) with every other parameter pinned to its values (legacy_reference_parameters());
//                        changing a pinned one is an error. It takes the merged frame alone of the geometry
//                        calibration: one rotation axis and one vector, with the recorded angles and heights.
//   filters              projection filter chain ("paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)", JSON, or
//                        "none"); TYPE[.k].param=VALUE changes one value of it (TYPE[k]: the k-th filter of the type)
//   backprojector        fbp(interp=, oversample=, filter_name=, filter_par=) (default) | ghbp | sirt(...) | mlem(...)
//                        | gridrec(...) | python(fn=...) (Python only)
//   geometry             "" (default): geometry.json in the cache directory, made there by the geometry calibration
//                        on first use or when it is other scans' | the path of one made before
//   row_time_us          for that step when no stretch holds a refill: the time of a stored row, microseconds (signed)
//   sharpness_weighting  "auto" (default: the intensity calibration's blur field, none if nothing significant) |
//                        "off" | "a11:a12:a21:a22:c1:c2"
//   exposure_arc         0 (default) | "auto" (the scan's recorded fraction) | a fraction; disabled with
//                        angle_interp=views
//   angle_subset         "" | "I/N": subset I of N interleaved subsets of the projections
//   sld                  "" | a scintillator light diffusion model (sld_params)
//   and the reconstructor's own (ReconConfig): margin, n_grid, angular_tolerance_steps, luminosity_repartition,
//   mend_threshold, log_clip_min, log_clip_max, block_angles, grid_size (0: 2 x axis rounded to even, + 1),
//   ghbp_legs, ghbp_reduction_steps, clip_outer_circle, axis_source_meters, x_interp, z_interp, angle_interp,
//   sharp_f, sharp_band.
// Older names: detector_smear (sharpness_weighting); removed with the calibration they belonged to, and refused by
// name: z_translation_scale, companion_registration, stationary_registration, axis_refinement, rolling_shutter.
#include <array>
#include <functional>
#include <optional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "helirec/calibration/geometry_calibration.hpp"
#include "helirec/calibration/intensity.hpp"
#include "helirec/calibration/scan.hpp"
#include "helirec/filters.hpp"
#include "helirec/geometry.hpp"
#include "helirec/pipeline.hpp"
#include "helirec/sld.hpp"

namespace helirec {

struct Json;

// ---------------------------------------------------------------- parameters

// The default parameters (mode "corrected"), as a JSON object's text
std::string default_parameters();
// The parameters mode legacy_reference pins (all but mode and the calibration's), as a JSON object's text
std::string legacy_reference_parameters();
// What names the calibration rather than the reconstruction: kept by every mode ("geometry", "row_time_us")
const std::vector<std::string>& calibration_keys();
// The conventions of a mode (provenance), as a JSON object's text
std::string mode_conventions(const std::string& mode);
// The uint8 window of the reference workflow, in the units of `mode` (cm^-1 for corrected: twice the reference's)
std::array<double, 2> display_window(const std::string& mode);

// A filter chain from its text ("paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)", a JSON list, "" or "none"),
// as a JSON list of {"type": ..., "name": value} (values: whole numbers, reals or text, as parse_value reads them)
std::string parse_filter_chain(const std::string& text);
// A chain (JSON list text) changed by {"TYPE[.k].param": value, "filters": a whole chain} (JSON object text): a
// new value of the first filter of the type (TYPE[k]: the k-th); an error when the chain has no such filter
std::string override_filter_chain(const std::string& chain, const std::string& changes);
// A chain (JSON list text) as its text form, numbers in %g
std::string filter_chain_text(const std::string& chain);
// "TYPE.param" or "TYPE[k].param" (or "filters"): a key of the filter chain
bool is_filter_key(const std::string& key);
// --set's reading of a value: a whole number, a real, or the text itself (JSON text of it)
std::string parse_value(const std::string& text);
// --set KEY=VALUE items as a JSON object of changes: values typed as the defaults' (booleans, whole numbers, reals,
// text); "filters" read as a chain, TYPE.param as parse_value
std::string parse_settings(const std::vector<std::string>& items);

// The parameters `params` (JSON object text) changed by `changes` (JSON object text): the rules of mode
// legacy_reference (pinned parameters; leaving it starts again from the defaults), filter keys applied to the chain,
// unknown and removed keys refused. Returns the new parameters' JSON text.
std::string update_parameters(const std::string& params, const std::string& changes);

// A scintillator light diffusion model: "" (none), "fraction=0.1,length=100[,bin=8,stride=10,outside=dark|lit]",
// "profile=bm18,fraction=0.06:0.03,length=40:200[,border_x=,border_y=,stride=]", "json=PATH" or a path (a Night
// Rail calibration: fraction and scale_l), or that as a JSON object (text). resolved: receives the model as the
// reconstructor's keys (sld_fraction, ... ; provenance).
SldParams sld_params(const std::string& spec, std::string* resolved = nullptr);

// "" -> (0, 1); "I/N" -> (I, N)
std::pair<int, int> parse_angle_subset(const std::string& spec);

// A backprojector spec (text or a JSON object's text) as the reconstructor's settings it names (a JSON object's
// text: backprojector, and bp_interp, bp_oversample, ramp_filter_name, ramp_filter_par, tomo_params or external_bp
// as the type takes them); unknown types and parameters are errors.
std::string backprojector_config(const std::string& spec);
// The same applied to a configuration. external: makes the backprojector
// of type "python" (its parameters as text); null: that type is refused.
using ExternalBackprojectorFactory = std::function<std::shared_ptr<ExternalBackprojector>(const FilterParams&)>;
void apply_backprojector(const std::string& spec, ReconConfig& cfg, const ExternalBackprojectorFactory& external = nullptr);

// ---------------------------------------------------------------- the reconstruction

struct SliceInfo {
    double fraction = 0;
    int view_height = 0;
    int nabu_index = 0;
    double height_mm = 0;
};

class Reconstruction {
  public:
    struct Options {
        std::string cache_dir;  // the calibration cache ("": none; then the geometry must be given)
        Device device = Device::CUDA;
        std::function<void(const std::string&)> log;
        bool progress = false;  // progress bars for the intensity calibration's reduction
        ExternalBackprojectorFactory external_backprojector;
    };
    // projections: the folder holding the two scans' folders; params: changes over the defaults (JSON object text)
    Reconstruction(const std::string& projections, const std::string& params, Options options);
    ~Reconstruction();

    // Change parameters (JSON object text): the reconstructor is made again; the calibration is kept unless the
    // geometry calibration named changes
    void set_parameters(const std::string& changes);
    std::string parameters() const;  // JSON object text

    const calibration::Scan& main() const;
    const calibration::Scan& companion() const;
    const calibration::IntensityCalibration& intensity() const;
    std::shared_ptr<SliceReconstructor> reconstructor() const;
    std::pair<int, int> doable() const;  // [lo, hi) view heights
    std::string mode() const;
    double rot_center() const;
    double rolling_shutter() const;       // the geometry calibration's row timing (0 in mode legacy_reference)
    double exposure_arc() const;          // the parameter resolved
    std::string sharpness_weighting() const;  // the parameter resolved: "a11:...:c2" or "" (off)
    std::string resolved_sld() const;     // the scintillator model as the reconstructor's keys (JSON object text)
    std::string info() const;             // the reconstructor's (JSON object text)

    // A fraction f of the doable view heights [lo, hi): h = hi - 1 - round_half_even(f (hi - lo)) (0 the bottom)
    SliceInfo slice_info(double fraction) const;
    // Everything that defines the reconstruction (JSON object text): mode and its conventions, the parameters
    // resolved, the calibrations, physics and the build
    std::string provenance() const;

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

// (path, text) of the geometry calibration a reconstruction uses: `given` (a path) as it is; else geometry.json of
// cache_dir, made there by the geometry calibration step (`make`, by default calibration::geometry_calibration::
// calibrate_geometry, with row_time_us if not "") when it is not there or names other scans than `masters`.
using GeometryMaker = std::function<std::string(const std::vector<std::string>& masters, const calibration::geometry_calibration::Options&)>;
std::pair<std::string, std::string> geometry_calibration_file(const std::string& given, const std::string& cache_dir, const std::vector<std::string>& masters,
                                                              const std::string& row_time_us, Device device,
                                                              const std::function<void(const std::string&)>& log = nullptr, const GeometryMaker& make = nullptr);

// ---------------------------------------------------------------- one tile of a mosaic
// A reconstructor of one tile alone (python/helirec/mosaic.py describes the tiles' description, geometry.hpp
// TileGeometry reads it): the detector map, the pose and the angle of every frame, the rows' timing, in the tile's
// own frame (the slice's centre is the tile's rotation axis, rounded to half a pixel of the reconstruction); only the
// helix frames on disk the description names are used. The intensity side is the pair's: the scan's own darks and
// flats taken to the nominal ring current, a weight map from the flats with a taper at the detector's far edge and
// across the axis.
struct TileOptions {
    Device device = Device::CUDA;
    std::string params = "{}";        // over the defaults here: fbp, no filters, no sld, no sharpness weighting
    // the slices' numbering (rows; the same for tiles that are compared); none: from the record's height of the tile's
    // middle frame on disk
    std::optional<double> row_origin;
    std::function<void(std::vector<float>& weight, int rows, int cols)> weight;  // changes the weight map (stored pixels)
    double cache_gb = 0;              // decoded chunks the reader keeps
    const calibration::DarksFlats* reduced = nullptr;  // the scan's darks and flats as reduced before (none: reduced here)
    std::string cache_dir;            // the darks and flats kept there (reduced_darks_flats)
    // 1, 2 or 3: at a half, a quarter, an eighth, from the codestream's own lower resolutions (RawReader(reduce=)),
    // the darks and flats through the same low-pass (at_resolution) and the whole geometry divided by 2^reduce
    int reduce = 0;
    std::function<void(const std::string&)> log;
    ExternalBackprojectorFactory external_backprojector;
};
struct TileReconstruction {
    std::shared_ptr<SliceReconstructor> reconstructor;
    double rot_center = 0;            // where the axis lies in the slice (reduced pixels)
    std::int64_t first_frame = 0, end_frame = 0;
    geometry::Terms terms;            // as given to the reconstructor (reduced)
    double row_origin = 0;            // stored rows
    calibration::DarksFlats reduced;  // the darks and flats at full resolution (to pass on to another call)
    int reduce = 0;
    calibration::Scan scan;
    std::string parameters;           // the parameters used (JSON object text)
};
TileReconstruction tile_reconstructor(const std::string& master, const std::string& description, const std::string& tile, const TileOptions& options);

// helirec's source revision (git describe of the source tree it was built from, if it is a git checkout)
std::string build_revision();

}  // namespace helirec
