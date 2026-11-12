// The checks in order, for one scan or for the two scans of an acquisition.
#include <cstdio>
#include <fstream>
#include <memory>

#include "helirec/calibration/verify.hpp"

namespace helirec::calibration {

namespace {

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

void write_file(const std::string& path, const std::string& text) {
    std::ofstream o(path);
    if (!o) throw Error("cannot write " + path);
    o << text;
}

}  // namespace

Report verify(const std::vector<std::string>& scans, const VerifyOptions& opt) {
    if (scans.empty()) throw Error("verify: no scan");
    std::vector<Record> records;
    for (const std::string& path : scans) records.push_back(read_record(path));
    // two scans that differ in the lateral stage of the rotation axis are the pair this was first written for;
    // anything else of two scans or more is checked as tiles (scans that differ in where the sample sits)
    const bool pair = records.size() == 2 && lateral_stage_differs(records[0], records[1]);
    const bool tiles = records.size() >= 2 && !pair;
    Report report;
    report.position_target_px = tolerances(records[0], opt.view.eps).eps;
    std::size_t printed = 0;
    const auto flush = [&] {
        if (opt.print) {
            Report part;
            part.findings.assign(report.findings.begin() + printed, report.findings.end());
            std::fputs(part.text().c_str(), stdout);
            std::fflush(stdout);
        }
        printed = report.findings.size();
        if (!opt.out_dir.empty()) {
            write_file(opt.out_dir + "/report.json", report.json());
            write_file(opt.out_dir + "/report.txt", report.text());
        }
    };

    if (has(opt.checks, "record")) {
        for (const Record& r : records) check_record(r, report, opt.view.eps, opt.radius_px);
        if (pair) check_record_pair(records[0], records[1], report, opt.view.eps);
        if (tiles) check_record_tiles(records, report, opt.view.eps, opt.radius_px);
        flush();
    }
    const bool frames_needed = has(opt.checks, "inventory") || has(opt.checks, "series") || has(opt.checks, "beam") || has(opt.checks, "views");
    if (!frames_needed) return report;

    std::vector<std::unique_ptr<Frames>> frames;
    for (const Record& r : records) frames.push_back(std::make_unique<Frames>(r, opt.threads));

    std::vector<Inventory> inventories(records.size());
    bool have_inventory = false;
    if (!opt.inventory_dir.empty()) {
        for (std::size_t k = 0; k < records.size(); ++k)
            inventories[k] = load_inventory(records[k], frames[k]->cols(), opt.inventory_dir + "/inventory_" + records[k].label);
        have_inventory = true;
    } else if (has(opt.checks, "inventory")) {
        for (std::size_t k = 0; k < records.size(); ++k) {
            inventories[k] = check_inventory(records[k], *frames[k], report, opt.inventory_step);
            if (!opt.out_dir.empty()) save_inventory(inventories[k], records[k], opt.out_dir + "/inventory_" + records[k].label);
            flush();
        }
        have_inventory = true;
    }
    std::vector<FlatField> flats;
    if (has(opt.checks, "series") || has(opt.checks, "beam") || has(opt.checks, "views"))
        for (std::size_t k = 0; k < records.size(); ++k) flats.push_back(make_flat_field(*frames[k]));
    if (has(opt.checks, "series")) {
        for (std::size_t k = 0; k < records.size(); ++k) {
            const bool with_other = k == 0 && pair;
            check_series(records[k], *frames[k], flats[k], report, with_other ? &records[1] : nullptr, with_other ? frames[1].get() : nullptr,
                         with_other ? &flats[1] : nullptr);
        }
        // tiles: the dark of every scan against the first's (8.1.2)
        for (std::size_t k = 1; tiles && k < records.size(); ++k) check_dark_pair(records[0], flats[0], records[k], flats[k], report);
        flush();
    }
    if (has(opt.checks, "beam") && have_inventory) {
        for (std::size_t k = 0; k < records.size(); ++k) check_beam(records[k], inventories[k], flats[k], report);
        flush();
    }
    if (has(opt.checks, "views") && tiles) {
        Finding& f = report.add("5 6", records[0].label + "+" + std::to_string(records.size() - 1), "views that record the same lines");
        f.found = "not checked here: the comparisons of a mosaic (one tile against another, opposite views and a turn apart within a tile) are placed by the caller (Session::measure)";
        f.verdict = "unverified";
        flush();
    } else if (has(opt.checks, "views")) {
        ViewOptions view = opt.view;
        if (view.table.empty() && !opt.out_dir.empty()) view.table = opt.out_dir + "/views.tsv";
        check_views(records[0], *frames[0], flats[0], pair ? &records[1] : nullptr, pair ? frames[1].get() : nullptr, pair ? &flats[1] : nullptr, view, report);
        flush();
    }
    return report;
}

}  // namespace helirec::calibration
