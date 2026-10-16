// Standalone native profiler workload: one raw 128^3 uint8 CT chunk.
#include "pipeline.hpp"
#include <fstream>
#include <iostream>

using namespace vcr::augment;

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: profile_augment RAW_CHUNK copy|flip|median|blur|affine|elastic|sheet|local|morphology|noise [iterations=5]\n";
        return 2;
    }
    const int iterations = argc == 4 ? std::stoi(argv[3]) : 5;
    if (iterations < 1) return 2;
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
    else if (name == "affine") transforms.push_back(Affine{.scale = 1 / 1.1});
    else if (name == "elastic") transforms.push_back(Elastic{.sigma = 3., .magnitude = 1.});
    else if (name == "sheet") transforms.push_back(SheetCompression{.strength = .2, .intensity_sigma = 1., .smoothing = 3., .axes = "z"});
    else if (name == "local") transforms.push_back(LocalGamma{.gamma = .8, .local = {.scale = 12., .loc = .5}});
    else if (name == "morphology") {
        for (std::size_t i = 0; i < image.size(); ++i) sample.at("image").array.set(i, image.get(i) > .5);
        transforms.push_back(Morphology{.radius = 1., .on = {"image"}, .repeated_cross = true});
    }
    else if (name == "noise") transforms.push_back(GaussianNoise{.std = .05});
    else if (name != "copy") return 2;
    Pipeline pipeline(transforms, {{"image", Target{}}});
    double checksum = 0;
    for (int i = 0; i < iterations; ++i) {
        auto result = pipeline.run(sample, 123, false, false);
        checksum += result.sample.at("image").array.get(0);
    }
    std::cout << checksum << '\n';
}
