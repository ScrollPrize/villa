#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <algorithm>
#include <iostream>
using namespace vc::objects;
int main() try
{
    TemporaryDirectory directory;
    OpenOptions open; open.author = "region-test";
    auto store = ObjectStore::create(directory.path / "region.vcobj", open);
    std::vector<std::string> ids;
    {
        auto txn = store.write();
        for (int i = 0; i < 3; ++i) {
            ObjectRecord record; record.kind = "fiber"; record.fiber.emplace(); record.fiber->length = 20;
            const auto id = store.upsertObject(txn, record); ids.push_back(id);
            vc::geometry::Fiber fiber; fiber.controlPoints = {{-10,0,0}, {10,0,0}};
            store.putGeometry(txn, id, vc::geometry::encode(fiber));
        }
        txn.commit();
    }
    std::sort(ids.begin(), ids.end());
    auto txn = store.read();
    FiberSlice slice;
    FiberRegionOptions options;
    auto all = store.fiberRegion(txn, slice, options);
    require(all.complete() && all.intersectingFibers == 3, "boundary crossing segments");
    options.maximumFibers = 1; options.selected = ids.front();
    options.uuids = std::vector<std::string>{ids.back()};
    auto filtered = store.fiberRegion(txn, slice, options);
    require(filtered.intersectingFibers == 1 && filtered.runs[0].objectUuid == ids.back(), "filter before selected and limit");
    require(filtered.runs[0].points == all.runs.back().points, "filter preserves exact geometry");
    require(!filtered.complete() && filtered.stop == FiberRegionStop::FiberLimit, "limit reported incomplete");
    options.uuids = std::vector<std::string>{};
    require(store.fiberRegion(txn, slice, options).complete() && store.fiberRegion(txn, slice, options).runs.empty(), "empty inclusion");
    require(store.fiberRegion(txn, slice, options, [](const auto&) { return true; }).stop == FiberRegionStop::Cancelled, "empty inclusion cancellation");
    options.uuids = std::vector<std::string>{ids.back(), ids.back()}; options.maximumFibers = 0;
    filtered = store.fiberRegion(txn, slice, options);
    require(filtered.complete() && filtered.intersectingFibers == 1, "duplicate inclusion no duplicate result");
    options.candidateWorkBudget = 0;
    require(store.fiberRegion(txn, slice, options).stop == FiberRegionStop::CandidateBudget, "filter setup obeys work budget");
    options.candidateWorkBudget = 1000000; options.processingByteBudget = 0;
    require(store.fiberRegion(txn, slice, options).stop == FiberRegionStop::ProcessingBudget, "filter setup obeys memory budget");
    options.processingByteBudget = 64*1024*1024; options.sqlStepBudget = 0;
    require(store.fiberRegion(txn, slice, options).stop == FiberRegionStop::SqlBudget, "filter setup obeys SQL budget");
    std::cout << "fiber region tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
