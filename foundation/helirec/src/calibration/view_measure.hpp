#pragma once
// The measurement of one pair of blocks that, according to the record, hold the same lines: cut from raw frames
// at whole frames, rows and columns (view_plan.hpp), converted to line integrals with the scan's own dark and
// mean flat (frames.hpp), and registered (registration.hpp), cutting B again by whole samples until less than a
// sample is left. check_views (verify_views.cpp) and measure_views (verify.hpp) both measure through this.
#include <memory>
#include <string>
#include <vector>

#include "calibration/view_plan.hpp"
#include "helirec/calibration/registration.hpp"
#include "helirec/calibration/verify.hpp"

namespace helirec::calibration {

struct ViewScan {
    const Record* rec;
    Frames* frames;
    const FlatField* ff;
};

// A block of n samples cut with a margin of m on both sides
struct Size {
    int n = 0, m = 0;
    int total() const { return n + 2 * m; }
};
// the largest n <= max_inner with a margin m >= min_margin for which n and n + 2 m are quick transform sizes and
// n + 2 m <= max_total
Size pick_size(int max_total, int max_inner, int min_margin);
Size diagnostic_columns(const Size& full, const ViewOptions& opt);

// what the record states per frame of a stretch: angle, and height in rows (row_sign: +1 if a larger axis-stage
// position puts the sample on higher rows)
Trajectory trajectory(const Record& rec, const Stretch& s, double row_sign);

struct Pairing {
    ViewScan A, B;
    int sa = 0, sb = 0;  // stretches
    const Trajectory* ta = nullptr;
    const Trajectory* tb = nullptr;
    double angle_offset = 0;
    bool mirror = false;
    // a model to be tested, applied to the blocks before they are registered (none in the view checks):
    const DetectorMap* map = nullptr;  // A's block is cut on the map's ideal grid, and B's too unless map_b is set
    const DetectorMap* map_b = nullptr;  // B's own map: what differs between two scans is tested on one of them
    double frame_shear = 0;            // B's frames are read at an angle that changes by this many frames per column
    double sway_px = 0, sway_phase_deg = 0;  // B's frames are read sway_px cos(angle - phase) columns further on (verify.hpp)
};

struct Cut {
    int row_a = 0, col_a = 0, col_b = 0;  // first row and column of A's block; first column of B's block (before reversal)
    Size rows, cols;
};

struct Measured {
    BlockPlan plan;   // as finally cut
    Cut cut;
    double d[3] = {0, 0, 0};  // registration: frames against the plan; rows and columns including whole re-cuts
    double correlation = 0;
    int evaluations = 0;
    bool ok = false;
};

// first row of A's block so that it and B's block, N rows higher and with its margins, lie centred in the detector
bool place_rows(int rows, int N, const Size& size, int& row_a);

// Registrars of the block sizes in use, most recent first: their plans, weights and work memory serve every block
// of the same size.
struct Registrars {
    Device device = Device::CPU;
    std::vector<std::unique_ptr<Registrar>> items;
    Registrar& get(const int n[3], const int margin[3], const RegistrationOptions& opt);
};
Registrars& registrars();

// where the time of the view checks goes [s]
struct ViewTiming {
    double read = 0, integrals = 0, registration = 0;
    int blocks = 0, registrations = 0;
    int attempted_views = 0, rejected_views = 0;
};
ViewTiming& view_timing();

// cut both blocks, register, and cut B again by whole samples until less than a sample is left
bool measure(const Pairing& P, const BlockPlan& first_plan, Cut cut, const RegistrationOptions& ropt, const ViewOptions& opt, Measured& m);

}  // namespace helirec::calibration
