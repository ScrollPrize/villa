// helirec-calibration: verify what an acquisition record states, against itself and against the raw frames.
//
//   helirec-calibration record <scan.h5> [<more scans.h5> ...] [--json FILE]
//   helirec-calibration verify <scan.h5> [<more scans.h5> ...] [--checks LIST] [--out DIR] [--inventory DIR]
//                              [--threads N] [--frames N] [--rows N] [--blocks N] [--position-target PX] [--radius PX]
//                              [--device cpu|cuda|auto]
//   helirec-calibration selftest [--device cpu|cuda|auto]
//
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
// Everything here is helirec::calibration::verify (helirec/calibration/verify.hpp).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "helirec/calibration/registration.hpp"
#include "helirec/calibration/verify.hpp"

using namespace helirec;
using namespace helirec::calibration;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  helirec-calibration record <scan.h5> [<more scans.h5> ...] [--json FILE]\n"
                 "  helirec-calibration verify <scan.h5> [<more scans.h5> ...] [--checks record,inventory,series,beam,views]\n"
                 "                             [--out DIR] [--inventory DIR] [--threads N] [--frames N] [--rows N] [--blocks N]\n"
                 "                             [--columns N (0: full)] [--column-fraction F (0..1)]\n"
                 "                             [--position-target PX (default 0.01)] [--radius PX] [--inventory-step N] [--device cpu|cuda|auto]\n"
                 "  helirec-calibration selftest [--device cpu|cuda|auto]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string command = argv[1];
    try {
        if (command != "record" && command != "verify" && command != "selftest") return usage();

        std::vector<std::string> scans;
        std::string json, device = "auto";
        VerifyOptions opt;
        opt.print = true;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            const auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw Error("option " + a + " needs a value");
                return argv[++i];
            };
            if (a == "--json") json = value();
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
        if (command == "record") opt.checks = "record";
        const Report report = verify(scans, opt);
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
