#include <vc_object_store/objects/ObjectStore.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
using namespace vc::objects;
// Read-only timing of the volume-targeted reads VC3D issues while viewing and
// editing fibers: a viewport region, a line-annotation strip's candidates, the
// unbound count, a catalog page and a curved view's control chains.
int main(int argc, char** argv) try {
    if (argc != 3 && argc != 6) {
        std::cerr << "Usage: bench_volume_queries STORE.vcobj VOLUME_ID [X Y Z]\n";
        return 2;
    }
    const std::string volume = argv[2];
    std::array<double, 3> c{16896.6, 16494.5, 47008.2};
    if (argc == 6) for (int i = 0; i < 3; ++i) c[i] = std::stod(argv[3 + i]);
    OpenOptions options; options.readOnly = true;
    auto db = ObjectStore::open(argv[1], options);
    auto time = [&](const char* name, const std::function<size_t(const ReadTxn&)>& read) {
        std::vector<double> ms;
        size_t items = 0;
        for (int i = 0; i < 5; ++i) {
            auto txn = db.read();
            const auto start = std::chrono::steady_clock::now();
            items = read(txn);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        std::sort(ms.begin(), ms.end());
        std::cout << name << " median_ms=" << ms[2] << " min_ms=" << ms.front() << " max_ms=" << ms.back()
                  << " items=" << items << '\n' << std::flush;
    };
    time("region_300x200", [&](const ReadTxn& txn) {
        RegionSlice slice; slice.origin = {c[0] - 150, c[1] - 100, c[2]}; slice.localBounds = {0, 0, -3, 300, 200, 3};
        RegionOptions region; region.volume = volume; region.maximumObjects = 5000;
        return db.geometryRegion(txn, slice, region).runs.size();
    });
    time("region_1500x1000", [&](const ReadTxn& txn) {
        RegionSlice slice; slice.origin = {c[0] - 750, c[1] - 500, c[2]}; slice.localBounds = {0, 0, -3, 1500, 1000, 3};
        RegionOptions region; region.volume = volume; region.maximumObjects = 5000;
        return db.geometryRegion(txn, slice, region).runs.size();
    });
    time("strip_spatial_regions", [&](const ReadTxn& txn) {
        ObjectSelection selection; selection.kind = "fiber"; selection.volume = volume;
        selection.bbox = std::array<double, 6>{c[0] - 200, c[1] - 200, c[2] - 2000, c[0] + 200, c[1] + 200, c[2] + 2000};
        return db.spatialRegions(txn, selection).size();
    });
    time("unbound_count", [&](const ReadTxn& txn) { return size_t(db.unboundObjectCount(txn, "fiber", volume)); });
    time("object_count", [&](const ReadTxn& txn) { return size_t(db.objectCount(txn, "fiber")); });
    time("catalog_page_200_with_unbound", [&](const ReadTxn& txn) {
        ObjectSelection page; page.kind = "fiber"; page.volume = volume; page.orderBy = "length_desc";
        page.includeUnbound = true; page.limit = 200;
        return db.selectObjects(txn, page, std::vector<std::string>{"name", "fiber"}).size();
    });
    time("chain_box_select", [&](const ReadTxn& txn) {
        ObjectSelection selection; selection.kind = "fiber"; selection.volume = volume; selection.limit = 5000;
        selection.bbox = std::array<double, 6>{c[0] - 500, c[1] - 500, c[2] - 3000, c[0] + 500, c[1] + 500, c[2] + 3000};
        return db.selectObjects(txn, selection, std::vector<std::string>{"fiber", "meta"}).size();
    });
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
