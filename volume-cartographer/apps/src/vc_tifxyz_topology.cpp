#include "vc_tifxyz_topology_impl.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/PointCollections.hpp"
#include "utils/Json.hpp"

#include <boost/program_options.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace po = boost::program_options;

using vc_topology::Census;
using vc_topology::Params;

namespace {

const char* const CLASSES[] = {"islands", "holes", "tears", "folds"};

bool has_class(const Census& c, const std::string& name)
{
    if (name == "islands") return c.island_quads > 0;
    if (name == "holes") return c.hole_count > 0;
    if (name == "tears") return c.tear_edges > 0;
    if (name == "folds") return c.fold_quads > 0;
    return false;
}

utils::Json xyz(const vc_topology::Vec3& p)
{
    utils::Json a = utils::Json::array();
    a.push_back(p.x);
    a.push_back(p.y);
    a.push_back(p.z);
    return a;
}

void census_json(const Census& c, utils::Json& j)
{
    j["grid_rows"] = c.rows;
    j["grid_cols"] = c.cols;
    j["valid_vertices"] = (int64_t)c.valid_vertices;
    j["valid_quads"] = (int64_t)c.valid_quads;
    j["isolated_vertices"] = (int64_t)c.isolated_vertices;
    j["area_vx2"] = c.area_vx2;
    j["median_step_u_vx"] = c.median_step_u;
    j["median_step_v_vx"] = c.median_step_v;

    utils::Json comps = utils::Json::array();
    for (const auto& k : c.components) {
        utils::Json r = utils::Json::object();
        r["quads"] = (int64_t)k.quads;
        r["area_vx2"] = k.area_vx2;
        r["grid"] = utils::Json::array();
        r["grid"].push_back(k.v);
        r["grid"].push_back(k.u);
        r["site"] = xyz(k.site);
        comps.push_back(std::move(r));
    }
    utils::Json islands = utils::Json::object();
    islands["components"] = (int64_t)c.component_count;
    islands["island_quads"] = (int64_t)c.island_quads;
    islands["island_area_vx2"] = c.island_area_vx2;
    islands["component_sites"] = std::move(comps);
    j["islands"] = std::move(islands);

    utils::Json hs = utils::Json::array();
    for (const auto& h : c.holes) {
        utils::Json r = utils::Json::object();
        r["cells"] = (int64_t)h.cells;
        r["grid_bbox"] = utils::Json::array();
        r["grid_bbox"].push_back(h.v0);
        r["grid_bbox"].push_back(h.u0);
        r["grid_bbox"].push_back(h.v1);
        r["grid_bbox"].push_back(h.u1);
        r["site"] = xyz(h.site);
        hs.push_back(std::move(r));
    }
    utils::Json holes = utils::Json::object();
    holes["count"] = (int64_t)c.hole_count;
    holes["cells"] = (int64_t)c.hole_cells;
    holes["hole_sites"] = std::move(hs);
    j["holes"] = std::move(holes);

    utils::Json ts = utils::Json::array();
    for (const auto& t : c.tears) {
        utils::Json r = utils::Json::object();
        r["edges"] = (int64_t)t.edges;
        r["max_len_vx"] = t.max_len;
        r["max_ratio"] = t.max_ratio;
        r["grid"] = utils::Json::array();
        r["grid"].push_back(t.v);
        r["grid"].push_back(t.u);
        r["site"] = xyz(t.site);
        ts.push_back(std::move(r));
    }
    utils::Json tears = utils::Json::object();
    tears["edges"] = (int64_t)c.tear_edges;
    tears["sites"] = (int64_t)c.tear_site_count;
    tears["max_ratio"] = c.max_edge_ratio;
    tears["tear_sites"] = std::move(ts);
    j["tears"] = std::move(tears);

    utils::Json fs = utils::Json::array();
    for (const auto& f : c.folds) {
        utils::Json r = utils::Json::object();
        r["quads"] = (int64_t)f.quads;
        r["grid"] = utils::Json::array();
        r["grid"].push_back(f.v);
        r["grid"].push_back(f.u);
        r["site"] = xyz(f.site);
        fs.push_back(std::move(r));
    }
    utils::Json folds = utils::Json::object();
    folds["quads"] = (int64_t)c.fold_quads;
    folds["sites"] = (int64_t)c.fold_site_count;
    folds["quads_on_unused_diagonal"] = (int64_t)c.fold_quads_other_diagonal;
    folds["degenerate_quads"] = (int64_t)c.degenerate_quads;
    folds["fold_sites"] = std::move(fs);
    j["folds"] = std::move(folds);
}

}  // namespace

int main(int argc, char** argv)
{
    po::options_description desc(
        "Census tifxyz surfaces for grid-topology defects: islands, holes, "
        "tears and folds.\nReport-only: nothing is modified");
    desc.add_options()
        ("help,h", "Print help")
        ("surface", po::value<std::vector<std::string>>()->multitoken(),
         "Input tifxyz surface directories; several may be given")
        ("output,o", po::value<std::string>(),
         "Output report file (.json)")
        ("collection", po::value<std::string>(),
         "Also write defect sites as a point collection (.json), loadable "
         "in VC3D. Sites from several surfaces are merged, so this is only "
         "meaningful for surfaces traced on the same volume")
        ("tear-factor", po::value<double>()->default_value(4.0),
         "A grid edge is a tear when it is longer than this multiple of the "
         "surface's own median step in that direction. Must be > 1")
        ("max-sites", po::value<int>()->default_value(200),
         "Cap on located sites written per defect class per surface; the "
         "counts above them are always complete")
        ("max-collection-points", po::value<int>()->default_value(10000),
         "Cap on overlay points written to --collection")
        ("fail-on", po::value<std::string>(),
         "Exit with code 3 if any of these classes is found, for use as a "
         "gate in scripts: a comma-separated list of islands, holes, tears, "
         "folds, or 'any'");

    po::positional_options_description pos;
    pos.add("surface", -1);

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

    const auto surfaces = vm["surface"].as<std::vector<std::string>>();
    const std::string output_path = vm["output"].as<std::string>();

    Params params;
    params.tear_factor = vm["tear-factor"].as<double>();
    params.max_sites = vm["max-sites"].as<int>();

    if (!(params.tear_factor > 1.0) || !std::isfinite(params.tear_factor)) {
        std::cerr << "Error: --tear-factor must be a finite number greater "
                     "than 1 (got " << params.tear_factor << ")" << std::endl;
        return 1;
    }
    if (params.max_sites < 0) {
        std::cerr << "Error: --max-sites must be >= 0 (got "
                  << params.max_sites << ")" << std::endl;
        return 1;
    }

    std::set<std::string> fail_on;
    if (vm.count("fail-on")) {
        std::stringstream ss(vm["fail-on"].as<std::string>());
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            if (tok == "any") {
                for (const char* k : CLASSES) fail_on.insert(k);
                continue;
            }
            bool known = false;
            for (const char* k : CLASSES) known |= (tok == k);
            if (!known) {
                std::cerr << "Error: --fail-on does not know the class '"
                          << tok << "'; it takes islands, holes, tears, "
                             "folds or any" << std::endl;
                return 1;
            }
            fail_on.insert(tok);
        }
    }

    const auto t0 = std::chrono::steady_clock::now();

    utils::Json entries = utils::Json::array();
    utils::Json flagged = utils::Json::array();
    std::vector<std::pair<std::string, Census>> done;
    int load_failures = 0;
    int64_t with[4] = {0, 0, 0, 0};
    bool gate_tripped = false;

    for (const std::string& path : surfaces) {
        utils::Json entry = utils::Json::object();
        entry["surface"] = path;
        std::unique_ptr<QuadSurface> surface;
        try {
            surface = load_quad_from_tifxyz(path);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load surface from " << path << ": "
                      << e.what() << std::endl;
            entry["error"] = std::string(e.what());
            entries.push_back(std::move(entry));
            ++load_failures;
            continue;
        }
        Census c;
        try {
            c = vc_topology::census(*surface->rawPointsPtr(), params);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        census_json(c, entry);
        entries.push_back(std::move(entry));

        utils::Json classes = utils::Json::array();
        for (int i = 0; i < 4; ++i)
            if (has_class(c, CLASSES[i])) {
                ++with[i];
                classes.push_back(std::string(CLASSES[i]));
                if (fail_on.count(CLASSES[i])) gate_tripped = true;
            }
        if (!classes.empty()) {
            utils::Json f = utils::Json::object();
            f["surface"] = path;
            f["classes"] = std::move(classes);
            f["island_quads"] = (int64_t)c.island_quads;
            f["hole_cells"] = (int64_t)c.hole_cells;
            f["tear_edges"] = (int64_t)c.tear_edges;
            f["fold_quads"] = (int64_t)c.fold_quads;
            flagged.push_back(std::move(f));
        }

        std::cerr << path << ": " << c.valid_quads << " quads, "
                  << c.component_count << " component(s), "
                  << c.island_quads << " island quad(s), " << c.hole_count
                  << " hole(s), " << c.tear_edges << " tear edge(s) in "
                  << c.tear_site_count << " site(s), " << c.fold_quads
                  << " folded quad(s) in " << c.fold_site_count << " site(s)"
                  << std::endl;
        done.emplace_back(path, std::move(c));
    }

    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cerr << "census wall time: " << wall << " s" << std::endl;

    utils::Json report = utils::Json::object();
    report["tool"] = "vc_tifxyz_topology";
    report["report_only"] = true;
    report["note"] = std::string(
        "Counts are over the surface as this codebase loads it: z <= 0 cells "
        "are invalid and mask.tif applies. 'islands' are valid quads outside "
        "the largest edge-connected component, which is the component "
        "vc_flatten keeps; 'holes' are runs of invalid cells enclosed by "
        "valid ones, not the padding around the sheet; 'tears' are grid "
        "edges longer than tear_factor times the surface's own median step "
        "in that direction; 'folds' are quads whose two triangles face "
        "opposite ways on either diagonal. Self-intersection is censused by "
        "vc_tifxyz_selfcross, not here.");
    utils::Json p = utils::Json::object();
    p["tear_factor"] = params.tear_factor;
    p["max_sites"] = params.max_sites;
    report["parameters"] = std::move(p);

    utils::Json summary = utils::Json::object();
    summary["surfaces"] = (int64_t)surfaces.size();
    summary["censused"] = (int64_t)done.size();
    summary["failed_to_load"] = (int64_t)load_failures;
    summary["with_islands"] = with[0];
    summary["with_holes"] = with[1];
    summary["with_tears"] = with[2];
    summary["with_folds"] = with[3];
    summary["flagged"] = std::move(flagged);
    report["summary"] = std::move(summary);
    report["surfaces"] = std::move(entries);

    std::ofstream o(output_path);
    if (!o.is_open()) {
        std::cerr << "Error: failed to open output file " << output_path
                  << std::endl;
        return 1;
    }
    o << report.dump(4);
    o.flush();
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
        const int cap = vm["max-collection-points"].as<int>();
        PointCollections coll;
        const std::string names[4] = {"topology-islands", "topology-holes",
                                      "topology-tears", "topology-folds"};
        const cv::Vec3f colors[4] = {{1.0f, 0.55f, 0.1f}, {0.2f, 0.6f, 1.0f},
                                     {1.0f, 0.1f, 0.1f}, {0.8f, 0.2f, 1.0f}};
        std::vector<cv::Vec3f> pts[4];
        int written = 0;
        auto add = [&](int slot, const vc_topology::Vec3& s) {
            if (written >= cap) return;
            pts[slot].emplace_back((float)s.x, (float)s.y, (float)s.z);
            ++written;
        };
        for (const auto& [path, c] : done) {
            for (size_t i = 1; i < c.components.size(); ++i)
                add(0, c.components[i].site);
            for (const auto& h : c.holes) add(1, h.site);
            for (const auto& t : c.tears) add(2, t.site);
            for (const auto& f : c.folds) add(3, f.site);
        }
        for (int i = 0; i < 4; ++i) {
            if (pts[i].empty()) continue;
            coll.setCollectionColor(coll.addCollection(names[i]), colors[i]);
            coll.addPoints(names[i], pts[i]);
        }
        const std::string coll_path = vm["collection"].as<std::string>();
        if (!coll.saveToJSON(coll_path)) {
            std::cerr << "Error: failed to write point collection to "
                      << coll_path << std::endl;
            return 1;
        }
        std::cout << "Point collection (" << written << " sites) written to "
                  << coll_path << std::endl;
    }

    std::cout << done.size() << " surface(s) censused: " << with[0]
              << " with islands, " << with[1] << " with holes, " << with[2]
              << " with tears, " << with[3] << " with folds." << std::endl;

    if (load_failures > 0) {
        std::cerr << load_failures << " surface(s) could not be read."
                  << std::endl;
        return 1;
    }
    if (gate_tripped) return 3;
    return 0;
}
