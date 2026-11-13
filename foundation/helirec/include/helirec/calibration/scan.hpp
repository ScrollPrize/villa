#pragma once
// A scan as the reconstruction reads it, from a BLISS tomo master file (ESRF BM18): where the frames of its
// projections, darks and flat series lie in the detector's files, and per projection the angle, the height and
// the ring current, per flat series the ring current of each frame and an angle, and the acquisition's scalars in
// SI units.
//
// Everything is derived as the reference conversion derives it (nxtomomill h52nx followed by nx2etf), and agrees
// with the files that conversion writes bit for bit:
//
//   Scan field                      reference (ETF projections.h5 and flats.h5)
//   projections (frame p)           projections.h5:/data (a virtual dataset), frame p
//   angles_deg                      framewise/angles_deg: the record's rotation, wrapped as BLISS stores it
//   unwrapped_angles_deg            framewise/unwrapped_angles_deg
//   z_translation_m                 framewise/z_translation_m
//   control_A                       framewise/control: the ring current of each projection [A]
//   flats_control_A                 the NX control/data of the flat frames
//   flats_unwrapped_angles_deg      flats.h5:/framewise/unwrapped_angles_deg
//   y_translation_m                 framewise/y_translation_m (constant)
//   estimated_cor_from_motor_pix    globals/estimated_cor_from_motor_pix
//   distance_m                      globals/distance_m (the effective propagation distance)
//   energy_kev, pixel_size_m        globals/beam_energy_kev, x_pixel_size_m
//
// The entries of the master are taken in nxtomomill's order (by start time, to the millisecond) and typed by its
// rules (technique/image_key, else technique/scan_category, else the title). The ring current of a frame is
// nxtomomill's: every entry from the tomo sequence's start on that records a current gives it at its start time
// (one value) or spread evenly from its start time to its end time (several), and each frame, spread likewise over
// its entry's times, takes the current interpolated linearly between those in time (in nanoseconds since the
// epoch, as doubles).
//
// helirec::calibration::Record is the same file read verbatim; this is the reading the reconstruction needs.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace helirec::calibration {

// Where a sequence of frames is stored: frame i is frame frame_index[i] of the dataset `dataset` in
// files[file_index[i]] (the detector's own files, not the master's virtual dataset).
struct FrameList {
    std::vector<std::string> files;  // absolute paths
    std::string dataset;
    std::vector<std::int32_t> file_index, frame_index;

    std::size_t size() const { return file_index.size(); }
    // the frames `idx` (with the same files)
    FrameList subset(const std::vector<std::int64_t>& idx) const;
    // several lists one after the other (their files merged, in the order they first appear)
    static FrameList concatenate(const std::vector<FrameList>& parts);
};

// An entry of the tomo sequence: its role ("dark", "flat", "proj", "align") and the frames it holds
struct ScanEntry {
    std::string name, role, title;
    std::int64_t n_frames = 0;
};

struct Scan {
    std::string name;       // the master's file name without ".h5" ("..._HA_0001")
    std::string master;     // absolute path of the master file
    std::string directory;  // the directory that holds it (and scanNNNN/ with the detector's files)
    int rows = 0, cols = 0;
    std::string dtype;  // of a stored frame ("uint16")
    FrameList projections, darks;
    std::vector<FrameList> flats;  // one per uninterrupted series of flat frames, in the order of acquisition

    std::vector<double> angles_deg;            // the record's rotation [deg], wrapped as stored
    std::vector<double> unwrapped_angles_deg;  // the same unwrapped (numpy.unwrap, period 360)
    std::vector<double> z_translation_m;       // the record's height of the sample stage [m]
    std::vector<double> control_A;             // ring current at each projection [A]
    std::vector<std::vector<double>> flats_control_A;  // ring current at each flat frame [A], per series
    std::vector<double> flats_unwrapped_angles_deg;    // per series: the angle the series is attached to
    std::vector<double> darks_control_A;               // ring current at each dark frame [A]

    double y_translation_m = 0;               // lateral position of the rotation stage [m]
    double estimated_cor_from_motor_pix = 0;  // y_translation_m / pixel_size_m: the axis from (cols - 1) / 2
    double energy_kev = 0;                    // NaN when not recorded
    double distance_m = 0;                    // effective propagation distance; NaN when not recorded
    double pixel_size_m = 0;                  // sample pixel size (columns and rows)
    std::optional<double> detector_pixel_size_m, count_time_s;
    // continuous rotation: the share of the angular step swept during an exposure (the recorded
    // <rotation>_delta / <rotation>_period, else exposure / (exposure + latency)); none for a step scan or unknown
    std::optional<double> exposure_arc;
    std::string detector;  // "det3"
    std::string sample_name, start_time, end_time;
    std::vector<ScanEntry> entries;  // of the tomo sequence, in the reference's order

    // every detector file the scan's frames are read from, each once: the projections', the darks', then the flats'
    std::vector<std::string> referenced_files() const;
    // throws when a detector file the scan reads is not on disk (its virtual frames would read as zeros)
    void check_files() const;
};

// `path`: the master file, the scan directory <name>_0001 (holding <name>_0001.h5), or a directory holding exactly
// one such scan directory.
Scan open_scan(const std::string& path);

// The tomo scans below `root` (the directories D holding D.h5 whose master starts a tomo sequence; the walk does
// not enter a scan's directory and goes `max_depth` levels down), sorted by name. Masters that cannot be read
// as a scan are left out, and, with require_files, so are scans whose detector files are not all on disk.
std::vector<Scan> discover_scans(const std::string& root, int max_depth = 3, bool require_files = false);

// The name of one scan of an extended field of view, "<base>_<kind>[_fc_|_cs_]_NNNN[_NNNN]" with kind HA, TA,
// TA2 ... TA9 (the rule of night_rail_bm18's analyse_tetra_and_crashes).
struct TetraName {
    std::string base, kind, suffix;
    int rank = 0;        // HA 0, TA 1, TA2 2, ...
    bool crash = false;  // a crash-postfixed name (_fc_ / _cs_)
};
std::optional<TetraName> parse_tetra_name(const std::string& name);

// The families of an extended field of view among `scans` (indices into it), each the main scan first.
// Scans whose names carry a kind (HA, TA, ...) are grouped by base and number and ordered HA, TA, TA2, ...
// (crash-postfixed names are left out). Scans without one are grouped by geometry: the same frame, pixel size,
// energy, number of projections and angular range, another lateral position of the stage; the main scan is the
// one with its axis nearest the detector's centre (smallest |estimated_cor_from_motor_pix|).
std::vector<std::vector<int>> group_companions(const std::vector<Scan>& scans);

// The one family among `scans` (throws when there are none or several, or when its members are not compatible):
// indices of the main scan first, then its companions.
std::vector<int> pair_companions(const std::vector<Scan>& scans);

}  // namespace helirec::calibration
