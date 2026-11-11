#pragma once
// Verification of what an acquisition record states.
//
// Two kinds of check, kept apart:
//  * check_record*: the record against itself (counts, clocks, raw encoder counts against the derived angles
//    and heights, commanded against recorded motion). No frame is read.
//  * check_inventory / check_series / check_views: the record against the raw frames. The frames are used as
//    stored; the only processing anywhere is a dark subtraction and a division by the mean flat, both computed
//    here from the scan's own dark and flat frames (frames.hpp).
//
// A finding names the requirement it bears on by its number in
// development_notes/calibration/00_reconstruction_requirements.md.
#include <string>
#include <utility>
#include <vector>

#include "helirec/calibration/frames.hpp"
#include "helirec/calibration/record.hpp"

namespace helirec::calibration {

struct Finding {
    std::string requirement;  // "5.4.5"
    std::string scan;         // "HA", "TA", "HA+TA"
    std::string quantity;
    std::string stated;       // what the record states
    std::string found;        // what the check found
    // "holds": checked, true to the tolerance given; "fails": checked, false; "open": the check could not decide;
    // "note": a value read or derived, with nothing to compare it against
    std::string verdict;
    std::string note;
    std::vector<std::pair<std::string, double>> numbers;  // the values behind `stated` and `found`
};

struct Report {
    std::vector<Finding> findings;
    Finding& add(std::string requirement, std::string scan, std::string quantity);
    std::string json() const;
    std::string text() const;
};

// Tolerances of section 3 of the requirements, from the positional tolerance eps (pixels)
struct Tolerances {
    double eps = 0.1;
    double R = 0;            // largest radius to reconstruct [px]
    double H = 0;            // detector rows
    double step_deg = 0;     // angle between frames
    double angle_deg() const { return eps / R * 57.29577951308232; }
    double angle_frames() const { return angle_deg() / step_deg; }
    double axis_px() const { return eps / 2; }
    double rows() const { return eps; }
    double travel_scale() const { return eps / H; }
};
Tolerances tolerances(const Record& rec, double eps = 0.1);

// ---- the record against itself
void check_record(const Record& rec, Report& report);
void check_record_pair(const Record& first, const Record& second, Report& report);

// ---- the record against the frames

// Every frame of every stretch is read once: the detector software's per-frame statistics are recomputed and
// compared (4.1), the raw value range and outliers are collected (8.1.4, 8.1.7), and the mean level of column
// strips is kept per frame for the beam checks (8.2).
struct Inventory {
    int strip_cols = 32;
    struct PerStretch {
        std::vector<double> sum, min, max;   // over the statistics rectangle, per frame
        std::vector<float> strips;           // (n_frames, n_strips): mean raw value of each column strip in the rectangle
        std::vector<std::uint16_t> frame_max;  // over the whole frame
        std::vector<std::int64_t> histogram;   // of every raw value of the stretch (65536 bins)
    };
    std::vector<PerStretch> stretches;
};
Inventory check_inventory(const Record& rec, Frames& frames, Report& report);
void save_inventory(const Inventory& inv, const Record& rec, const std::string& path);  // <path>.bin + <path>.json
Inventory load_inventory(const Record& rec, int cols, const std::string& path);         // what save_inventory wrote

// Darks and flats: their roles, levels and noise (4.2, 8.1.1, 8.1.2 with a second scan, 8.2.5)
void check_series(const Record& rec, Frames& frames, const FlatField& ff, Report& report,
                  const Record* other = nullptr, Frames* other_frames = nullptr, const FlatField* other_ff = nullptr);

// The beam against the record: air level per frame from the inventory's strips (4.7, 8.2.1 - 8.2.5)
void check_beam(const Record& rec, const Inventory& inv, const FlatField& ff, Report& report);

struct ViewOptions {
    int frames = 32;        // frames per block
    int rows = 256;         // rows per block (fewer where the overlap is smaller)
    int blocks = 8;         // blocks along each stretch
    double eps = 0.1;
};
// Views that record the same lines, registered against each other (registration.hpp): opposite views of a turn
// at rest, a turn at rest against the helix, the helix a turn apart, and one scan against the other.
void check_views(const Record& first, Frames& first_frames, const FlatField& first_ff, const Record* second,
                 Frames* second_frames, const FlatField* second_ff, const ViewOptions& opt, Report& report);

}  // namespace helirec::calibration
