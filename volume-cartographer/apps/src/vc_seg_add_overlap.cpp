#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using json = nlohmann::json;

using Point3 = bg::model::point<float, 3, bg::cs::cartesian>;
using Box3 = bg::model::box<Point3>;
using Entry = std::pair<Box3, std::filesystem::path>;
using SegmentTree = bgi::rtree<Entry, bgi::quadratic<16>>;


int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <tgt-dir> <single-tiffxyz>" << std::endl;
        std::cout << "   this will check for overlap between any tiffxyz in target dir and <single-tiffxyz> and add overlap metadata" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path tgt_dir = argv[1];
    std::filesystem::path seg_dir = argv[2];
    int search_iters = 10;
    srand(clock());

    QuadSurface current(seg_dir);

    // Read existing overlapping data for current segment
    std::set<std::string> current_overlapping = read_overlapping_json(current.path);

    // Phase 1: Build R-tree index of segment bounding boxes
    SegmentTree segmentIndex;

    for (const auto& entry : std::filesystem::directory_iterator(tgt_dir)) {
        if (!std::filesystem::is_directory(entry))
            continue;

        std::string name = entry.path().filename();
        if (name == current.id)
            continue;

        std::filesystem::path meta_fn = entry.path() / "meta.json";
        if (!std::filesystem::exists(meta_fn))
            continue;

        std::ifstream meta_f(meta_fn);
        json meta;
        try {
            meta = json::parse(meta_f);
        } catch (const json::exception& e) {
            std::cerr << "Error parsing meta.json for " << name << ": " << e.what() << std::endl;
            continue;
        }

        if (!meta.count("bbox"))
            continue;

        if (meta.value("format","NONE") != "tifxyz")
            continue;

        // Extract bbox from meta.json (format: [[low_x,low_y,low_z], [high_x,high_y,high_z]])
        auto bbox_json = meta["bbox"];
        Box3 box(
            Point3(bbox_json[0][0], bbox_json[0][1], bbox_json[0][2]),
            Point3(bbox_json[1][0], bbox_json[1][1], bbox_json[1][2])
        );

        segmentIndex.insert({box, entry.path()});
    }

    std::cout << "Built R-tree index with " << segmentIndex.size() << " segments" << std::endl;

    // Phase 2: Query R-tree and check only candidates with intersecting bboxes
    Rect3D currentBbox = current.bbox();
    Box3 queryBox(
        Point3(currentBbox.low[0], currentBbox.low[1], currentBbox.low[2]),
        Point3(currentBbox.high[0], currentBbox.high[1], currentBbox.high[2])
    );

    std::vector<Entry> candidates;
    segmentIndex.query(bgi::intersects(queryBox), std::back_inserter(candidates));

    std::cout << "Found " << candidates.size() << " candidates with intersecting bboxes" << std::endl;

    bool found_overlaps = false;

    for (const auto& [box, seg_path] : candidates) {
        std::filesystem::path meta_fn = seg_path / "meta.json";
        std::ifstream meta_f(meta_fn);
        json meta = json::parse(meta_f);

        QuadSurface other(seg_path, meta);

        // Build SurfacePatchIndex for surface B to accelerate pointTo()
        SurfacePatchIndex patchIndex;
        std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
        surfaces.emplace_back(SurfacePatchIndex::SurfacePtr(&other, [](QuadSurface*){}));
        patchIndex.rebuild(surfaces, 10.0f);  // 10.0 bbox padding

        if (overlap(current, other, search_iters, &patchIndex)) {
            found_overlaps = true;

            // Add to current's overlapping set
            current_overlapping.insert(other.id);

            // Read and update other's overlapping set
            std::set<std::string> other_overlapping = read_overlapping_json(other.path);
            other_overlapping.insert(current.id);
            write_overlapping_json(other.path, other_overlapping);

            std::cout << "Found overlap: " << current.id << " <-> " << other.id << std::endl;
        }
    }

    // Write current's overlapping data
    if (found_overlaps || !current_overlapping.empty()) {
        write_overlapping_json(current.path, current_overlapping);
        std::cout << "Updated overlapping data for " << current.id
                  << " (" << current_overlapping.size() << " overlaps)" << std::endl;
    }

    return EXIT_SUCCESS;
}
