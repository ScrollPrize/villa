// Standalone native profiler workload: one raw 128^3 uint8 CT chunk.
#include "pipeline.hpp"
#include <fstream>
#include <iostream>

using namespace vcr::augment;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: profile_augment RAW_CHUNK copy|flip|median|blur\n";
        return 2;
    }
    Array image({1, 128, 128, 128});
    std::vector<unsigned char> bytes(image.size());
    std::ifstream input(argv[1], std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
        std::cerr << "expected a complete raw 128^3 uint8 chunk\n";
        return 2;
    }
    for (std::size_t i = 0; i < bytes.size(); ++i) image.set(i, bytes[i] / 255.0);
    Sample sample{{"image", Field{image, Target{}, "image"}}};
    std::vector<Op> transforms;
    const std::string name = argv[2];
    if (name == "flip") transforms.push_back(Flip{.axes = "x", .p_axis = 1});
    else if (name == "median") transforms.push_back(MedianFilter{.size = 3.});
    else if (name == "blur") transforms.push_back(GaussianBlur{.sigma = 1.});
    else if (name != "copy") return 2;
    Pipeline pipeline(transforms, {{"image", Target{}}});
    for (int i = 0; i < 5; ++i) {
        auto result = pipeline.run(sample, 123, false, false);
        std::cout << result.sample.at("image").array.get(0) << '\n';
    }
}
