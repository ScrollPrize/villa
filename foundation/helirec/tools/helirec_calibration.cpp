// helirec-calibration: the calibration of an acquisition.
//
//   helirec-calibration check-record <scan.h5> [<more scans.h5> ...] [--no-frames] [--checks LIST] [--out DIR]
//                                    [--inventory DIR] [--json FILE] [--threads N] [--frames N] [--rows N] [--blocks N]
//                                    [--position-target PX] [--radius PX] [--device cpu|cuda|auto]
//   helirec-calibration geometry <scan.h5> [<more scans.h5> ...] --out FILE [--row-time-us US] [--order N] [--patch N]
//                                [--threads N] [--device cpu|cuda|auto]
//   helirec-calibration intensity <scan> <companion> --geometry FILE --out DIR [--no-progress]
//   helirec-calibration selftest [--device cpu|cuda|auto]
//
// intensity makes the intensity calibration of a two-scan acquisition in the frame its geometry calibration (FILE,
// made by `geometry`) merges the scans in: the darks and flats reduced, the weight maps, the detector's blur field
// and the placement of both scans. It writes calibration_arrays.bin and calibration.json to DIR, which the
// reconstruction takes as its cache directory, and reuses what is there when it is of the same acquisition. The
// scans: master files or scan directories, in the order the geometry calibration has them. It is
// helirec::calibration::calibrate_intensity (helirec/calibration/intensity.hpp).
//
// geometry measures the detector's map and the places of the scans of one sample from their own frames (a scan
// alone, a pair moved sideways, the tiles of a mosaic; a scan with the rotation axis inside the detector's field
// first) and writes them as JSON, which the reconstruction reads for a pair (helirec/geometry.hpp). --row-time-us:
// the time of a stored row within the frame, signed, if known for this camera configuration (0: not known, done
// without); measured from a refill otherwise. --order N: of the detector map's polynomial (default 3; 0: no map).
// --patch N: the plan's blocks are cut into patches of N x N (default 128), each registered once. It is
// helirec::calibration::geometry_calibration::calibrate_geometry (helirec/calibration/geometry_calibration.hpp).
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
// Everything of check-record is helirec::calibration::run_record_checks (helirec/calibration/record_checks.hpp).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "helirec/calibration/geometry_calibration.hpp"
#include "helirec/calibration/intensity.hpp"
#include "helirec/calibration/scan.hpp"
#include "helirec/calibration/registration.hpp"
#include "helirec/calibration/record_checks.hpp"

using namespace helirec;
using namespace helirec::calibration;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  helirec-calibration check-record <scan.h5> [<more scans.h5> ...] [--no-frames] [--checks record,inventory,series,beam,views]\n"
                 "                                   [--out DIR] [--inventory DIR] [--json FILE] [--threads N] [--frames N] [--rows N] [--blocks N]\n"
                 "                                   [--columns N (0: full)] [--column-fraction F (0..1)]\n"
                 "                                   [--position-target PX (default 0.01)] [--radius PX] [--inventory-step N] [--device cpu|cuda|auto]\n"
                 "  helirec-calibration geometry <scan.h5> [<more scans.h5> ...] --out FILE [--row-time-us US] [--order N] [--patch N]\n"
                 "                               [--threads N] [--device cpu|cuda|auto]\n"
                 "  helirec-calibration intensity <scan> <companion> --geometry FILE --out DIR [--no-progress]\n"
                 "  helirec-calibration selftest [--device cpu|cuda|auto]\n");
    return 2;
}

// helirec-calibration geometry: the arguments after the command
int run_geometry_calibration(int argc, char** argv) {
    namespace gc = helirec::calibration::geometry_calibration;
    std::vector<std::string> scans;
    std::string out, device = "auto";
    gc::Options opt;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw Error("option " + a + " needs a value");
            return argv[++i];
        };
        if (a == "--out") out = value();
        else if (a == "--row-time-us") opt.row_time = std::stod(value()) * 1e-6;
        else if (a == "--order") opt.order = std::stoi(value());
        else if (a == "--patch") opt.patch = std::stoi(value());
        else if (a == "--threads") opt.threads = std::stoi(value());
        else if (a == "--device") device = value();
        else if (!a.empty() && a[0] == '-') throw Error("unknown option " + a);
        else scans.push_back(a);
    }
    if (scans.empty() || out.empty()) return usage();
    if (device != "cpu" && device != "cuda" && device != "auto") throw Error("--device is cpu, cuda or auto");
    opt.device = device == "cuda" || (device == "auto" && cuda_available()) ? Device::CUDA : Device::CPU;
    opt.log = [](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
    };
    const std::string cal = gc::calibrate_geometry(scans, opt);
    gc::save_text(cal, out);
    std::printf("written to %s\n", out.c_str());
    return 0;
}

std::string base_name(const std::string& p) {
    const std::size_t i = p.find_last_of('/');
    return i == std::string::npos ? p : p.substr(i + 1);
}

// helirec-calibration intensity: the arguments after the command
int run_intensity_calibration(int argc, char** argv) {
    std::vector<std::string> paths;
    std::string geometry_path, out;
    bool progress = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw Error("option " + a + " needs a value");
            return argv[++i];
        };
        if (a == "--geometry") geometry_path = value();
        else if (a == "--out") out = value();
        else if (a == "--no-progress") progress = false;
        else if (!a.empty() && a[0] == '-') throw Error("unknown option " + a);
        else paths.push_back(a);
    }
    if (paths.size() != 2 || geometry_path.empty() || out.empty()) return usage();
    const auto log = [](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
    };
    const Scan main = open_scan(paths[0]), companion = open_scan(paths[1]);
    const geometry::Calibration geo = geometry::Calibration::load(geometry_path);
    const std::vector<std::string> made = geo.scans();
    if (made.size() != 2 || base_name(made[0]) != base_name(main.master) || base_name(made[1]) != base_name(companion.master))
        throw Error("the geometry calibration " + geometry_path + " was made from other scans (or in the other order)");
    const std::array<std::vector<double>, 2> heights = {geo.heights_rows(main.z_translation_m, main.pixel_size_m),
                                                        geo.heights_rows(companion.z_translation_m, companion.pixel_size_m)};
    const auto poses = geo.frame_poses({main.unwrapped_angles_deg, companion.unwrapped_angles_deg}, heights);
    const geometry::MergedLayout layout = geo.merged_layout(poses, heights);
    const IntensityCalibration cal = calibrate_intensity(main, companion, layout, out, log, progress);
    std::printf("written to %s (calibration_arrays.bin, calibration.json)\n", out.c_str());
    (void)cal;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string command = argv[1];
    try {
        if (command == "geometry") return run_geometry_calibration(argc, argv);
        if (command == "intensity") return run_intensity_calibration(argc, argv);
        if (command != "check-record" && command != "selftest") return usage();

        std::vector<std::string> scans;
        std::string json, device = "auto";
        RecordCheckOptions opt;
        opt.print = true;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            const auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw Error("option " + a + " needs a value");
                return argv[++i];
            };
            if (a == "--json") json = value();
            else if (a == "--no-frames") opt.checks = "record";
            else if (a == "--out") opt.out_dir = value();
            else if (a == "--checks") opt.checks = value();
            else if (a == "--inventory") opt.inventory_dir = value();
            else if (a == "--threads") opt.threads = std::atoi(value().c_str());
            else if (a == "--frames") opt.view.frames = std::atoi(value().c_str());
            else if (a == "--rows") opt.view.rows = std::atoi(value().c_str());
            else if (a == "--columns") opt.view.columns = std::stoi(value());
            else if (a == "--column-fraction") opt.view.column_fraction = std::stod(value());
            else if (a == "--blocks") opt.view.blocks = std::atoi(value().c_str());
            else if (a == "--position-target") {
                const auto v = value();
                std::size_t used = 0;
                opt.view.eps = std::stod(v, &used);
                if (used != v.size()) throw Error("--position-target must be a finite positive number");
            }
            else if (a == "--radius") opt.radius_px = std::stod(value());
            else if (a == "--inventory-step") opt.inventory_step = std::stoi(value());
            else if (a == "--device") device = value();
            else if (!a.empty() && a[0] == '-') throw Error("unknown option " + a);
            else scans.push_back(a);
        }
        if (device != "cpu" && device != "cuda" && device != "auto") throw Error("--device is cpu, cuda or auto");
        opt.view.device = device == "cuda" || (device == "auto" && cuda_available()) ? Device::CUDA : Device::CPU;
        if (command == "selftest") return registration_selftest(true, opt.view.device) ? 0 : 1;
        if (scans.empty()) return usage();
        const Report report = run_record_checks(scans, opt);
        if (!json.empty()) {
            std::ofstream o(json);
            if (!o) throw Error("cannot write " + json);
            o << report.json();
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "helirec-calibration: %s\n", e.what());
        return 1;
    }
}
