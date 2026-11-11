// helirec-calibration: verify what an acquisition record states, against itself and against the raw frames.
//
//   helirec-calibration record <scan.h5> [<second scan.h5>] [--json FILE]
//   helirec-calibration verify <scan.h5> [<second scan.h5>] [--checks LIST] [--out DIR] [--threads N]
//                              [--frames N] [--rows N] [--blocks N]
//   helirec-calibration selftest
//
// The first scan is the one with the rotation axis inside the detector's field. LIST is a comma-separated
// subset of record,inventory,series,beam,views (default: all; beam needs inventory).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "helirec/calibration/frames.hpp"
#include "helirec/calibration/record.hpp"
#include "helirec/calibration/registration.hpp"
#include "helirec/calibration/verify.hpp"

using namespace helirec;
using namespace helirec::calibration;

namespace {

void write_file(const std::string& path, const std::string& text) {
    std::ofstream o(path);
    if (!o) throw Error("cannot write " + path);
    o << text;
}

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  helirec-calibration record <scan.h5> [<second scan.h5>] [--json FILE]\n"
                 "  helirec-calibration verify <scan.h5> [<second scan.h5>] [--checks record,inventory,series,beam,views]\n"
                 "                             [--out DIR] [--threads N] [--frames N] [--rows N] [--blocks N]\n"
                 "  helirec-calibration selftest\n");
    return 2;
}

bool has(const std::string& list, const char* name) {
    std::size_t i = 0;
    while (i <= list.size()) {
        std::size_t j = list.find(',', i);
        if (j == std::string::npos) j = list.size();
        if (list.compare(i, j - i, name) == 0) return true;
        i = j + 1;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string command = argv[1];
    try {
        if (command == "selftest") return registration_selftest(true) ? 0 : 1;

        std::vector<std::string> scans;
        std::string json, out_dir, checks = "record,inventory,series,beam,views";
        int threads = 0;
        ViewOptions view;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            const auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw Error("option " + a + " needs a value");
                return argv[++i];
            };
            if (a == "--json") json = value();
            else if (a == "--out") out_dir = value();
            else if (a == "--checks") checks = value();
            else if (a == "--threads") threads = std::atoi(value().c_str());
            else if (a == "--frames") view.frames = std::atoi(value().c_str());
            else if (a == "--rows") view.rows = std::atoi(value().c_str());
            else if (a == "--blocks") view.blocks = std::atoi(value().c_str());
            else if (!a.empty() && a[0] == '-') throw Error("unknown option " + a);
            else scans.push_back(a);
        }
        if (scans.empty() || scans.size() > 2) return usage();

        std::vector<Record> records;
        for (const std::string& path : scans) records.push_back(read_record(path));
        Report report;

        if (command == "record") {
            for (const Record& r : records) check_record(r, report);
            if (records.size() == 2) check_record_pair(records[0], records[1], report);
            std::fputs(report.text().c_str(), stdout);
            if (!json.empty()) write_file(json, report.json());
            return 0;
        }
        if (command != "verify") return usage();

        const auto save = [&] {
            if (out_dir.empty()) return;
            write_file(out_dir + "/report.json", report.json());
            write_file(out_dir + "/report.txt", report.text());
        };
        std::size_t printed = 0;
        const auto flush = [&] {
            Report part;
            part.findings.assign(report.findings.begin() + printed, report.findings.end());
            printed = report.findings.size();
            std::fputs(part.text().c_str(), stdout);
            std::fflush(stdout);
            save();
        };

        if (has(checks, "record")) {
            for (const Record& r : records) check_record(r, report);
            if (records.size() == 2) check_record_pair(records[0], records[1], report);
            flush();
        }
        std::vector<std::unique_ptr<Frames>> frames;
        for (const Record& r : records) frames.push_back(std::make_unique<Frames>(r, threads));

        std::vector<Inventory> inventories(records.size());
        if (has(checks, "inventory")) {
            for (std::size_t k = 0; k < records.size(); ++k) {
                inventories[k] = check_inventory(records[k], *frames[k], report);
                if (!out_dir.empty()) save_inventory(inventories[k], records[k], out_dir + "/inventory_" + records[k].label);
                flush();
            }
        }
        std::vector<FlatField> flats;
        if (has(checks, "series") || has(checks, "beam") || has(checks, "views"))
            for (std::size_t k = 0; k < records.size(); ++k) flats.push_back(make_flat_field(*frames[k]));
        if (has(checks, "series")) {
            for (std::size_t k = 0; k < records.size(); ++k)
                check_series(records[k], *frames[k], flats[k], report, k == 0 && records.size() == 2 ? &records[1] : nullptr,
                             k == 0 && records.size() == 2 ? frames[1].get() : nullptr, k == 0 && records.size() == 2 ? &flats[1] : nullptr);
            flush();
        }
        if (has(checks, "beam") && has(checks, "inventory")) {
            for (std::size_t k = 0; k < records.size(); ++k) check_beam(records[k], inventories[k], flats[k], report);
            flush();
        }
        if (has(checks, "views")) {
            check_views(records[0], *frames[0], flats[0], records.size() == 2 ? &records[1] : nullptr, records.size() == 2 ? frames[1].get() : nullptr,
                        records.size() == 2 ? &flats[1] : nullptr, view, report);
            flush();
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "helirec-calibration: %s\n", e.what());
        return 1;
    }
}
