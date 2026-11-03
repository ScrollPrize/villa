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
        std::cout << "Native array, sampling and geometry APIs passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
