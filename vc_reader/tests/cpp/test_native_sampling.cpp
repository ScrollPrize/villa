#include <vc_reader/array.hpp>
#include <vc_reader/sampling.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

int main()
{
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() / ("vcr-native-array-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{path};
    try {
        fs::create_directories(path);
        std::array<uint16_t, 8> data{1000, 1002, 1004, 1006, 1008, 1010, 1012, 1014};
        { std::ofstream output(path / "0.0.0", std::ios::binary);
          output.write(reinterpret_cast<const char*>(data.data()), sizeof(data)); }
        vcr::ArraySpec spec;
        spec.shape = spec.chunks = {2, 2, 2};
        spec.itemsize = 2;
        spec.fill = {0, 0};
        spec.identity = path.string();
        spec.layers.push_back({"array", path.string(), "", "", false});
        vcr::Array reader(spec, std::make_shared<vcr::MemoryCache>(1024));
        std::array<uint16_t, 8> region{};
        reader.read_many({{0, 0, 0}}, {{2, 2, 2}}, {reinterpret_cast<uint8_t*>(region.data())});
        require(region == data, "uint16 region read");
        const double point[3]{.5, .5, .5};
        float value = 0;
        reader.sample_linear(point, 1, {'u', 2, false}, false, &value);
        require(value == 1007.f, "uint16 trilinear sample");
        const int64_t integerPoint[3]{1, 1, 1};
        uint16_t nearest = 0;
        reader.read_points(integerPoint, 1, reinterpret_cast<uint8_t*>(&nearest));
        require(nearest == 1014, "point read");

        vcr::aligned::Plan aligned({2,2,2}, {1,1,1}, {1,1,1}, {0,0,0}, {0,0,0},
            {.5f,.5f,.5f}, {1,0,0,0,1,0,0,0,1}, {2,2,2}, 1, 1, false, true);
        bool valid = false;
        uint16_t fill = 0;
        aligned.cpu({data.data()}, &fill, {'u',2,false}, &value, &valid, {});
        require(valid && value == 1007.f, "aligned sampling");
        const float vertices[]{0,0,0, 2,0,0};
        const int64_t offsets[]{0,2};
        vcr::geometry::Curves curves(vertices, 2, offsets, 2, nullptr);
        float output[3]{};
        vcr::geometry::interpolate(curves.view(false), 0, 1.f, output, &valid);
        require(valid && output[0] == 1.f && output[1] == 0.f, "curve interpolation");

        const float near_points[]{10.5f, 20.f, 30.f, NAN, 0.f, 0.f, 12.f, 21.f, 33.f};
        const auto origin = vcr::pointOrigin(near_points, 3);
        require(origin && *origin == std::array<std::int64_t, 3>{10, 20, 30}, "point origin");
        const double dispersed[]{0., 0., 0., 5000., 0., 0.};
        require(!vcr::pointOrigin(dispersed, 2), "dispersed points have no shared origin");
        const float none[]{NAN, NAN, NAN};
        require(*vcr::pointOrigin(none, 1) == std::array<std::int64_t, 3>{0, 0, 0}, "no finite point");

        const auto local = vcr::localPoints(near_points, 1);
        require(!local.dispersed && local.base == std::array<std::int64_t, 3>{10, 20, 30} && local.local[0] == .5f, "local points");
        require(vcr::localPoints(dispersed, 2).dispersed, "dispersed points stay global");
        const auto given = vcr::localPoints(near_points, 1, vcr::Index{1, 2, 3});
        require(given.base == std::array<std::int64_t, 3>{1, 2, 3} && given.local[0] == 10.5f, "a given origin");
        const double crop[]{5.25, 40.5, -3.5};
        const auto placed = vcr::localOrigin(crop);
        require(placed.base == std::array<std::int64_t, 3>{5, 40, 0} && placed.origin[0] == .25f && placed.origin[2] == -3.5f, "local origin");
        bool invalid = false;
        try { vcr::checkedCropShape({1, 1}); } catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "crop shape rank");
        const double start[]{55.25, 2., -1.};
        const std::int64_t volume[]{200, 200, 200};
        std::int64_t patch[3];
        float offset[3], center[4];
        bool supported;
        vcr::sheet_normal::region(start, volume, patch, offset, &supported);
        require(patch[2] == 0 && patch[1] == -54 && offset[0] == 1.25f && offset[2] == 2.f && !supported, "sheet normal region");
        vcr::sheet_normal::center(offset, supported, center);
        require(center[0] == 49.25f && center[3] == 0.f, "sheet normal center");

        const auto moved = vcr::aligned::make_field({1,0,0,0,1,0,0,0,1}, {1,2,3}, {2,1,0}, false, false);
        require(moved.enabled && moved.component[0] == 2 && moved.translation[2] == 3.f, "aligned field");
        bool rejected = false;
        try { vcr::aligned::make_field({}, {}, {0,0,1}, false, false); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "field components must permute XYZ");

        namespace geo = vcr::geometry;
        const std::int64_t ids[]{0, 0}, missing_id[]{5};
        const float arcs[]{0.f, 2.f};
        float points_out[6], gathered_arcs[2], distance[1], nearest_arc[1], plane_arc[1];
        std::int64_t owners[2], index[1], owner[1];
        bool valids[2];
        geo::gather_batch(curves.view(false), -1, ids, false, points_out, gathered_arcs, owners, valids, 2);
        require(valids[0] && points_out[0] == 0.f && owners[1] == 0, "gather batch");
        geo::interpolate_batch(curves.view(false), -1, ids, arcs, points_out, valids, 2);
        require(valids[1] && points_out[3] == 2.f, "interpolate batch");
        geo::interpolate_batch(curves.view(false), -1, missing_id, arcs, points_out, valids, 1);
        require(!valids[0], "interpolate batch reports a missing curve");
        const float query[]{1.f, 1.f, 0.f}, x_plane[]{1.f, 0.f, 0.f, 1.f};
        geo::nearest_batch(curves.view(false), -1, query, nullptr, 10.f, points_out, distance, nearest_arc, index, owner, valids, 1);
        require(valids[0] && distance[0] == 1.f && points_out[0] == 1.f, "nearest batch");
        geo::intersect_batch(curves.view(false), -1, ids, x_plane, nullptr, 0, points_out, plane_arc, index, valids, 1);
        require(valids[0] && plane_arc[0] == 1.f, "intersect batch");
        const float shift[]{1,0,0,5, 0,1,0,6, 0,0,1,7};
        geo::transform_batch(-1, query, shift, 1, false, points_out, 1);
        require(points_out[0] == 6.f && points_out[2] == 7.f, "transform batch");
        geo::transform_batch(-1, query, shift, 1, true, points_out, 1);
        require(points_out[0] == 1.f, "vectors skip translation");
        const auto grid = geo::voxel_grid({2, 2, 2}, 1);
        const float voxel[]{1.f, 0.f, 0.f};
        const geo::Voxels voxels(voxel, nullptr, 1, grid);
        bool inside = false;
        geo::contains_batch(voxels, voxel, nullptr, &inside, 1);
        require(inside, "contains batch");
        std::int64_t keys[2];
        geo::copy_keys(voxels, keys, 2);
        require(keys[1] == geo::missing_key, "keys pad with the missing key");
        std::uint8_t dense[8];
        geo::fill_dense(voxels, dense);
        require(dense[0] + dense[1] + dense[2] + dense[3] + dense[4] + dense[5] + dense[6] + dense[7] == 1, "dense voxels");
        std::cout << "Native array, sampling and geometry APIs passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
