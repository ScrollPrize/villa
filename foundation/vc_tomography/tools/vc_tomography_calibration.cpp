// vc_tomography_calibration: the calibration of an acquisition.
//
//   vc_tomography_calibration check-record <scan.h5> [<more scans.h5> ...] [--no-frames] [--checks LIST] [--out DIR]
//                                    [--inventory DIR] [--json FILE] [--threads N] [--frames N] [--rows N] [--blocks N]
//                                    [--position-target PX] [--radius PX] [--device cpu|cuda|auto]
//   vc_tomography_calibration geometry <scan.h5> [<more scans.h5> ...] --out FILE [--row-time-us US] [--rows-from-middle] [--order N | --radial-map] [--patch N] [--reduce N] [--projections STORE.zarr]
//                                [--threads N] [--device cpu|cuda|auto]
//   vc_tomography_calibration intensity <scan> <scan> [<more scans> ...] --geometry FILE --out DIR [--no-progress]
//   vc_tomography_calibration selftest [--device cpu|cuda|auto]
//
// intensity makes the intensity calibration of scans moved sideways (the scan with the rotation axis in its field
// first) in the frame its geometry calibration (FILE,
// made by `geometry`) merges the scans in: the darks and flats reduced, the weight maps, the detector's blur field
// and every scan's placement. It writes calibration_arrays.bin and calibration.json to DIR, which the
// reconstruction takes as its cache directory, and reuses what is there when it is of the same acquisition. The
// scans: master files or scan directories, in the order the geometry calibration has them. It is
// vc_tomography::calibration::calibrate_intensity (vc_tomography/calibration/intensity.hpp).
//
// geometry measures the detector's map and the places of the scans of one sample from their own frames (a scan
// alone, a pair moved sideways, the tiles of a mosaic; a scan with the rotation axis inside the detector's field
// first) and writes them as JSON, which the reconstruction reads for a pair (vc_tomography/geometry.hpp). --row-time-us:
// the time of a stored row within the frame, signed, if known for this camera configuration (0: not known, done
// without); measured from a refill otherwise. --rows-from-middle: the camera's two halves are read at once, the row
// time counted from the middle row to both ends; a row time not given is then measured from the frames between
// scans at different heights where no refill gives it. --order N: of the detector map's polynomial (default 3; 0: no map). --radial-map: the map's
// four terms of a lens' radial distortion solved as one unknown (a map of the third order): for scans compared
// along the frames' edges alone.
// --patch N: the plan's blocks are cut into patches of N x N (default 128), each registered once. It is
// vc_tomography::calibration::geometry_calibration::calibrate_geometry (vc_tomography/calibration/geometry_calibration.hpp).
//
// check-record checks what the acquisition record states, against itself and against the raw frames; it writes
// findings and changes nothing. --no-frames: the record against itself alone (the same as --checks record).
// One scan; or two scans that differ in the lateral stage of the rotation axis, the one with the axis inside the
// detector's field first; or the tiles of a mosaic (scans that differ in where the sample sits on the rotation
// stage), the reference tile first. The view checks cover the first two; a mosaic's comparisons are placed
// through the Python session. --radius: the lever arm of the angle targets where it is not the detector's width
// (default for tiles: from their layout). --inventory-step N: the inventory reads one frame in N (1: every frame;
// default: every frame, but one in 16 of a stretch that is stored compressed and has no recorded statistics to
// compare its frames with). LIST is a comma-separated
// subset of record,inventory,series,beam,views (default: all; beam needs the inventory). The inventory reads
// every frame; --out DIR keeps it (inventory_<scan>.bin) and --inventory DIR takes it from there instead.
// The blocks of the view checks are registered on the GPU when there is one (--device auto, the default).
// Everything of check-record is vc_tomography::calibration::run_record_checks (vc_tomography/calibration/record_checks.hpp).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "vc_tomography/calibration/geometry_calibration.hpp"
#include "vc_tomography/calibration/intensity.hpp"
#include "vc_tomography/calibration/scan.hpp"
#include "vc_tomography/calibration/registration.hpp"
#include "vc_tomography/calibration/record_checks.hpp"
#include "io/command_line.hpp"
#include "io/paths.hpp"

using namespace vc_tomography;
using namespace vc_tomography::calibration;
using command_line::Args;
using command_line::say;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  vc_tomography_calibration check-record <scan.h5> [<more scans.h5> ...] [--no-frames] [--checks record,inventory,series,beam,views]\n"
                 "                                   [--out DIR] [--inventory DIR] [--json FILE] [--threads N] [--frames N] [--rows N] [--blocks N]\n"
                 "                                   [--columns N (0: full)] [--column-fraction F (0..1)]\n"
                 "                                   [--position-target PX (default 0.01)] [--radius PX] [--inventory-step N] [--device cpu|cuda|auto]\n"
                 "  vc_tomography_calibration geometry <scan.h5> [<more scans.h5> ...] --out FILE [--row-time-us US] [--rows-from-middle] [--order N | --radial-map] [--patch N] [--reduce N] [--projections STORE.zarr]\n"
                 "                               [--threads N] [--device cpu|cuda|auto]\n"
                 "  vc_tomography_calibration intensity <scan> <scan> [<more scans> ...] --geometry FILE --out DIR [--no-progress]\n"
                 "  vc_tomography_calibration selftest [--device cpu|cuda|auto]\n");
    return 2;
}

// vc_tomography_calibration geometry: the arguments after the command
int run_geometry_calibration(int argc, char** argv) {
    namespace gc = vc_tomography::calibration::geometry_calibration;
    const command_line::Spec spec{{{"out", 1, 's'}, {"row_time_us", 1, 'f'}, {"rows_from_middle", 0, 'b'}, {"order", 1, 'i'}, {"radial_map", 0, 'b'},
                                   {"patch", 1, 'i'}, {"reduce", 1, 'i'}, {"projections", 1, 's'}, {"threads", 1, 'i'}, {"device", 1, 's'}},
                                  {}, "scans", true};
    const Args a{command_line::parse(argc - 1, argv + 1, spec)};
    const std::vector<std::string> scans = a.texts("scans");
    const std::string out = a.text("out");
    if (scans.empty() || out.empty()) return usage();
    gc::Options opt;
    if (a.get("row_time_us")) opt.row_time = a.number("row_time_us") * 1e-6;
    opt.rows_from_middle = a.flag("rows_from_middle");
    if (a.get("order")) opt.order = static_cast<int>(a.number("order"));
    opt.radial_map = a.flag("radial_map");
    if (a.get("patch")) opt.patch = static_cast<int>(a.number("patch"));
    if (a.get("reduce")) opt.reduce = static_cast<int>(a.number("reduce"));
    if (a.get("projections")) opt.projections = a.text("projections");
    if (a.get("threads")) opt.threads = static_cast<int>(a.number("threads"));
    opt.device = parse_device(a.get("device") ? a.text("device") : "auto", true);
    opt.log = say;
    const std::string cal = gc::calibrate_geometry(scans, opt);
    gc::save_text(cal, out);
    std::printf("written to %s\n", out.c_str());
    return 0;
}

std::string base_name(const std::string& p) { return io::basename(p); }

// vc_tomography_calibration intensity: the arguments after the command
int run_intensity_calibration(int argc, char** argv) {
    const command_line::Spec spec{{{"geometry", 1, 's'}, {"out", 1, 's'}, {"no_progress", 0, 'b'}}, {}, "scans", true};
    const Args a{command_line::parse(argc - 1, argv + 1, spec)};
    const std::vector<std::string> paths = a.texts("scans");
    const std::string geometry_path = a.text("geometry"), out = a.text("out");
    const bool progress = !a.flag("no_progress");
    if (paths.size() < 2 || geometry_path.empty() || out.empty()) return usage();
    const auto log = say;
    std::vector<Scan> scans;
    for (const std::string& path : paths) scans.push_back(open_scan(path));
    const geometry::Calibration geo = geometry::Calibration::load(geometry_path);
    const std::vector<std::string> made = geo.scans();
    bool same = made.size() == scans.size();
    for (std::size_t k = 0; same && k < scans.size(); ++k) same = base_name(made[k]) == base_name(scans[k].master);
    if (!same) throw Error("the geometry calibration " + geometry_path + " was made from other scans (or in another order)");
    std::vector<std::vector<double>> angles, heights;
    std::vector<const Scan*> all;
    for (const Scan& sc : scans) {
        angles.push_back(sc.unwrapped_angles_deg);
        heights.push_back(geo.heights_rows(sc.z_translation_m, sc.pixel_size_m));
        all.push_back(&sc);
    }
    const geometry::MergedLayout layout = geo.merged_layout(geo.scan_poses(angles, heights), heights);
    calibrate_intensity(all, layout, out, log, progress);
    std::printf("written to %s (calibration_arrays.bin, calibration.json)\n", out.c_str());
    return 0;
}

// vc_tomography_calibration check-record / selftest: the arguments after the command
int run_record_checks_command(const std::string& command, int argc, char** argv) {
    const command_line::Spec spec{{{"json", 1, 's'}, {"no_frames", 0, 'b'}, {"out", 1, 's'}, {"checks", 1, 's'}, {"inventory", 1, 's'},
                                   {"threads", 1, 'i'}, {"frames", 1, 'i'}, {"rows", 1, 'i'}, {"columns", 1, 'i'}, {"column_fraction", 1, 'f'},
                                   {"blocks", 1, 'i'}, {"position_target", 1, 'f'}, {"radius", 1, 'f'}, {"inventory_step", 1, 'i'}, {"device", 1, 's'}},
                                  {}, "scans", true};
    const Args a{command_line::parse(argc - 1, argv + 1, spec)};
    const auto whole = [&](const char* k, int& v) {
        if (a.get(k)) v = static_cast<int>(a.number(k));
    };
    RecordCheckOptions opt;
    opt.print = true;
    if (a.flag("no_frames")) opt.checks = "record";
    if (a.get("out")) opt.out_dir = a.text("out");
    if (a.get("checks")) opt.checks = a.text("checks");
    if (a.get("inventory")) opt.inventory_dir = a.text("inventory");
    whole("threads", opt.threads);
    whole("frames", opt.view.frames);
    whole("rows", opt.view.rows);
    whole("columns", opt.view.columns);
    whole("blocks", opt.view.blocks);
    whole("inventory_step", opt.inventory_step);
    if (a.get("column_fraction")) opt.view.column_fraction = a.number("column_fraction");
    if (a.get("position_target")) opt.view.eps = a.number("position_target");
    if (a.get("radius")) opt.radius_px = a.number("radius");
    opt.view.device = parse_device(a.get("device") ? a.text("device") : "auto", true);
    if (command == "selftest") return registration_selftest(true, opt.view.device) ? 0 : 1;
    const std::vector<std::string> scans = a.texts("scans");
    if (scans.empty()) return usage();
    const Report report = run_record_checks(scans, opt);
    if (a.get("json")) io::write_text(a.text("json"), report.json(), false);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string command = argv[1];
    return command_line::tool_main("vc_tomography_calibration", [&] {
        if (command == "geometry") return run_geometry_calibration(argc, argv);
        if (command == "intensity") return run_intensity_calibration(argc, argv);
        if (command == "check-record" || command == "selftest") return run_record_checks_command(command, argc, argv);
        if (command == "-h" || command == "--help") throw command_line::Exit{"help"};
        return usage();
    }, usage);
}
