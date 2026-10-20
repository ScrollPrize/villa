#define VC_TEST_MAIN
#include "vc_test.hpp"
#include "vc_object_store/TrackCrossings.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

TEST_CASE("track crossing kernel: clustered medians, clearance and tie breaking")
{
    namespace tc = vc::track_crossings;
    std::vector<int32_t> xyz;
    for (int track = 0; track < 2; ++track)
        for (int i = 0; i <= 20; ++i) xyz.insert(xyz.end(), {0, track, i});
    const std::vector<int64_t> offsets{0, 21, 42};
    const std::vector<uint64_t> ids{17, uint64_t{1} << 40};
    tc::EventBuffer events{{{0,1,2,2}, {0,1,3,3}, {0,1,8,8}, {0,1,12,12}, {0,1,19,19}}};
    auto result = tc::consolidate_crossing_events(events, xyz, offsets, ids, 2);
    CHECK(events.size() == 0);
    // Clusters [2,3], [8,12], [19]; use upper median 12 in the middle.
    CHECK(result.accepted_events == 3);
    CHECK(result.paired_tracks == 2);
    CHECK(result.offsets == std::vector<int64_t>{0,1,2});
    CHECK(result.partners == std::vector<int32_t>{1,0});
    CHECK(result.self_local == std::vector<int32_t>{12,12});
    CHECK(result.positions == std::vector<double>{12,12});
    CHECK(result.clearances == std::vector<double>{8,8});
    tc::Parameters p;
    p.clusterIndexRadius = 0;
    tc::EventBuffer tied{{{0,1,8,8}, {0,1,12,12}}};
    auto tie = tc::consolidate_crossing_events(tied, xyz, offsets, ids, 1, {}, p);
    CHECK(tie.self_local == result.self_local);
}


TEST_CASE("track crossing radix sort is stable with constrained OpenMP teams")
{
    const std::vector<uint64_t> keys{9,1,9,3,1,(uint64_t{1}<<60)-1,0};
    const std::vector<uint32_t> expected{6,1,4,3,0,2,5};
#ifdef _OPENMP
    const auto dynamic = omp_get_dynamic();
    omp_set_dynamic(1);
#endif
    CHECK(vc::track_crossings::parallel_argsort(keys, 3) == expected);
#ifdef _OPENMP
    omp_set_dynamic(dynamic);
#endif
    CHECK(vc::track_crossings::parallel_argsort(keys, 1) == expected);
}


TEST_CASE("track crossing kernel rejects invalid array indices")
{
    namespace tc = vc::track_crossings;
    const std::vector<int32_t> coordinates{0,0,0};
    const std::vector<int64_t> offsets{0,1};
    const std::vector<uint64_t> keys{0}, ids{0};
    const std::vector<int8_t> families{0};
    const std::vector<uint32_t> invalidOrder{1};
    CHECK_THROWS(tc::scan_crossing_events(coordinates, offsets, families, keys, invalidOrder, 1));
    tc::EventBuffer invalidEvents{{{0,1,0,0}}};
    CHECK_THROWS(tc::consolidate_crossing_events(invalidEvents, coordinates, offsets, ids, 1));
    const std::vector<uint64_t> oversizedKey{uint64_t{1} << 60};
    CHECK_THROWS(tc::parallel_argsort(oversizedKey, 1));
}
