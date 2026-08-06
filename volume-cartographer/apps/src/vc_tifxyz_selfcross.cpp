// vc_tifxyz_selfcross: report where a tifxyz surface passes through itself.
//
// An embedded surface cannot pass through itself, however tightly the wraps
// are packed, so a transverse self-intersection in a trace is a defect with
// no innocent reading -- unlike proximity, which in a crushed scroll can be
// arbitrarily small and still correct. This tool censuses one surface's
// triangles for non-adjacent transverse contacts and reports them; it never
// modifies anything.
//
// Contacts are classified rather than collapsed to yes/no:
//
//   transverse  the two triangle interiors genuinely pass through each other
//   coplanar    (near) coplanar with overlapping projections -- two sheets
//               pressed flat look like this, so it is reported, not counted
//               as a crossing
//   grazing     the sign or interval data the test relies on is numerically
//               ambiguous AND the triangles come within the touch tolerance
//               of each other; ambiguity about a pair that never comes near
//               contact is not reported at all
//
// Adjacency is excluded by grid index: quads within --exclude cells of each
// other in both v and u share vertices, and their triangles sharing space is
// what a surface does. Everything beyond that window is nonlocal.
//
// Each quad is split into triangles both ways and the two censuses are
// reported separately, because a twisted quad can cross under one
// triangulation and not the other.
//
// The census runs on the surface as this codebase loads it: z <= 0 cells are
// invalid and mask.tif applies. The same method run on a raw tifxyz that
// counts z <= 0 cells as valid can differ slightly (measured 1.49% of contact
// rows across one 185-trace corpus, with no clean/not-clean verdict changed).
//
// The census kernel lives in vc_tifxyz_selfcross_impl.hpp so the regression
// tests (core/test/test_tifxyz_selfcross.cpp) run the identical code.
//
// Exit codes: 0 = census ran (regardless of what it found), 1 = error,
// 3 = --fail-on-crossing was given and transverse contacts were found.

#include "vc_tifxyz_selfcross_impl.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/PointCollections.hpp"
#include "utils/Json.hpp"

#include <boost/program_options.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace po = boost::program_options;

using vc_selfcross::Contact;
using vc_selfcross::CensusCounts;
using vc_selfcross::TRANSVERSE;

int main(int argc, char** argv)
{
    po::options_description desc(
        "Census a tifxyz surface for non-adjacent transverse "
        "self-intersections.\nReport-only: nothing is modified");
    desc.add_options()
        ("help,h", "Print help")
        ("surface", po::value<std::string>(),
         "Input tifxyz surface directory")
        ("output,o", po::value<std::string>(),
         "Output report file (.json)")
        ("collection", po::value<std::string>(),
         "Also write crossing sites as a point collection (.json), "
         "loadable in VC3D")
        ("exclude", po::value<int>()->default_value(1),
         "Chebyshev grid-index adjacency exclusion; 1 = shared-vertex "
         "neighbours")
        ("maxedge", po::value<double>()->default_value(60.0),
         "Drop quads with any edge longer than this (voxels); a triangle "
         "built across a grid discontinuity crosses everything it passes "
         "through. 0 disables")
        ("cell", po::value<double>()->default_value(40.0),
         "Broad-phase cell size (voxels); affects speed, never contact "
         "verdicts or counts (bucket ranges are expanded by the touch "
         "tolerance)")
        ("threads", po::value<int>()->default_value(0),
         "Worker threads; 0 = hardware concurrency")
        ("max-collection-points", po::value<int>()->default_value(10000),
         "Cap on overlay points written to --collection")
        ("fail-on-crossing",
         "Exit with code 3 if any transverse contact is found, for use "
         "as a gate in scripts");

    po::positional_options_description pos;
    pos.add("surface", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(desc).positional(pos).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (vm.count("help") || !vm.count("surface") || !vm.count("output")) {
        std::cout << desc << std::endl;
        return vm.count("help") ? 0 : 1;
    }

    const std::string surface_path = vm["surface"].as<std::string>();
    const std::string output_path = vm["output"].as<std::string>();
    const int exclude = vm["exclude"].as<int>();
    const double maxedge = vm["maxedge"].as<double>();
    const double cell = vm["cell"].as<double>();
    int nthreads = vm["threads"].as<int>();
    if (nthreads <= 0) nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads <= 0) nthreads = 4;

    // The census refuses these too; checking here turns them into a usage
    // message rather than an exception trace.
    if (!(cell > 0.0) || !std::isfinite(cell)) {
        std::cerr << "Error: --cell must be a finite positive number of "
                     "voxels (got " << cell << ")" << std::endl;
        return 1;
    }
    if (exclude < 0) {
        std::cerr << "Error: --exclude must be >= 0 (got " << exclude << ")"
                  << std::endl;
        return 1;
    }
    if (maxedge < 0 || !std::isfinite(maxedge)) {
        // NaN fails both < 0 and > 0, which would silently disable the
        // filter rather than complain.
        std::cerr << "Error: --maxedge must be a finite number >= 0, with 0 "
                     "disabling the filter (got " << maxedge << ")"
                  << std::endl;
        return 1;
    }

    std::unique_ptr<QuadSurface> surface;
    try {
        surface = load_quad_from_tifxyz(surface_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to load surface from " << surface_path
                  << ": " << e.what() << std::endl;
        return 1;
    }
    const cv::Mat_<cv::Vec3f>& P = *surface->rawPointsPtr();

    const auto t0 = std::chrono::steady_clock::now();

    // Both triangulations, reported separately. A twisted quad can cross
    // under one diagonal and not the other; a surface is only called clean
    // here when both censuses find nothing transverse.
    std::vector<Contact> contacts[2];
    CensusCounts counts[2];
    for (int d = 0; d < 2; ++d) {
        try {
            counts[d] = vc_selfcross::census(P, d, cell, exclude, maxedge,
                                             nthreads, contacts[d]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        std::cerr << "diagonal " << d << ": " << counts[d].triangles
                  << " triangles, " << counts[d].pairs_tested
                  << " pairs tested, " << counts[d].transverse
                  << " transverse, " << counts[d].coplanar << " coplanar, "
                  << counts[d].grazing << " grazing" << std::endl;
    }
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    const size_t transverse_total = counts[0].transverse
                                  + counts[1].transverse;

    utils::Json report = utils::Json::object();
    report["tool"] = "vc_tifxyz_selfcross";
    report["report_only"] = true;
    report["surface"] = surface_path;
    report["clean_of_transverse_self_intersection"] =
        (transverse_total == 0);
    report["note"] = std::string(
        "'clean' means zero non-adjacent transverse contacts under both quad "
        "triangulations, on the surface as loaded here (z <= 0 invalid, mask "
        "applied). Coplanar and grazing contacts are reported but are not "
        "crossings: two sheets pressed flat against each other are coplanar.");
    utils::Json params = utils::Json::object();
    params["exclude"] = exclude;
    params["maxedge"] = maxedge;
    // Recorded for provenance even though contact counts are independent of
    // it: pairs_tested legitimately varies with the broad-phase cell size.
    params["cell"] = cell;
    params["touch_tolerance"] = vc_selfcross::TOUCH_TOL;
    params["diagonals"] = utils::Json::array();
    params["diagonals"].push_back(0);
    params["diagonals"].push_back(1);
    report["parameters"] = params;
    report["grid_rows"] = P.rows;
    report["grid_cols"] = P.cols;
    // Timing goes to stderr, not into the report: with it there, two runs on
    // the same surface would produce different bytes, and a byte-reproducible
    // report is worth more than an embedded stopwatch.
    std::cerr << "census wall time: " << wall << " s" << std::endl;

    utils::Json diags = utils::Json::array();
    for (int d = 0; d < 2; ++d) {
        utils::Json jd = utils::Json::object();
        jd["diagonal"] = d;
        jd["triangles"] = (uint64_t)counts[d].triangles;
        jd["quads_dropped_for_edge_length"] = (uint64_t)counts[d].quads_dropped;
        jd["pairs_tested"] = (uint64_t)counts[d].pairs_tested;
        jd["transverse"] = (uint64_t)counts[d].transverse;
        jd["coplanar"] = (uint64_t)counts[d].coplanar;
        jd["grazing"] = (uint64_t)counts[d].grazing;
        utils::Json rows = utils::Json::array();
        for (const Contact& c : contacts[d]) {
            if (c.verdict != TRANSVERSE)
                continue;
            utils::Json r = utils::Json::object();
            r["quad1"] = utils::Json::array();
            r["quad1"].push_back(c.v1); r["quad1"].push_back(c.u1);
            r["quad2"] = utils::Json::array();
            r["quad2"].push_back(c.v2); r["quad2"].push_back(c.u2);
            r["tri1"] = c.t1; r["tri2"] = c.t2;
            r["penetration_vx"] = (double)c.penetration;
            r["angle_deg"] = (double)c.angle_deg;
            // The midpoint of the two triangles' shared intersection
            // segment: a point both interiors contain, not a centroid
            // average that can sit off the crossing for long triangles.
            r["site"] = utils::Json::array();
            r["site"].push_back(c.site.x);
            r["site"].push_back(c.site.y);
            r["site"].push_back(c.site.z);
            rows.push_back(std::move(r));
        }
        jd["transverse_contacts"] = std::move(rows);
        diags.push_back(std::move(jd));
    }
    report["census"] = std::move(diags);

    std::ofstream o(output_path);
    if (!o.is_open()) {
        std::cerr << "Error: failed to open output file " << output_path
                  << std::endl;
        return 1;
    }
    o << report.dump(4);
    o.flush();
    // A full disk or quota failure surfaces here, not at open(). Claiming
    // "Report written" and exiting 0 on a failed write would defeat the one
    // job a report-only tool has.
    if (!o.good()) {
        std::cerr << "Error: failed while writing " << output_path
                  << std::endl;
        return 1;
    }
    o.close();
    if (o.fail()) {
        std::cerr << "Error: failed to finish writing " << output_path
                  << std::endl;
        return 1;
    }
    std::cout << "Report written to " << output_path << std::endl;

    if (vm.count("collection")) {
        // Written through PointCollections itself, so the file is readable
        // wherever point collections are.
        const int cap = vm["max-collection-points"].as<int>();
        PointCollections coll;
        const std::string name = "selfcross-transverse";
        coll.addCollection(name);
        coll.setCollectionColor(coll.getCollectionId(name),
                                cv::Vec3f(1.0f, 0.1f, 0.1f));
        std::vector<cv::Vec3f> pts;
        for (int d = 0; d < 2 && (int)pts.size() < cap; ++d)
            for (const Contact& c : contacts[d]) {
                if (c.verdict != TRANSVERSE) continue;
                if ((int)pts.size() >= cap) break;
                pts.emplace_back((float)c.site.x, (float)c.site.y,
                                 (float)c.site.z);
            }
        coll.addPoints(name, pts);
        const std::string coll_path = vm["collection"].as<std::string>();
        if (!coll.saveToJSON(coll_path)) {
            std::cerr << "Error: failed to write point collection to "
                      << coll_path << std::endl;
            return 1;
        }
        std::cout << "Point collection (" << pts.size()
                  << " sites) written to " << coll_path << std::endl;
    }

    std::cout << (transverse_total == 0
        ? "No non-adjacent transverse self-intersection found."
        : "Surface has non-adjacent transverse self-intersections: "
          + std::to_string(counts[0].transverse) + " contacts on diagonal 0, "
          + std::to_string(counts[1].transverse) + " on diagonal 1.")
        << std::endl;

    if (transverse_total > 0 && vm.count("fail-on-crossing"))
        return 3;
    return 0;
}
