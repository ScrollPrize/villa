#include <vc_reader/transforms.hpp>
#include <iostream>

int main()
{
    using namespace vcr::transforms;
    try {
        const Target image{};
        Sample input;
        input["image"] = {Array({1, 3, 4}, 'f', 4), image};
        auto* values = static_cast<float*>(input.at("image").array.data());
        std::iota(values, values + 12, 1.f);
        BrightnessMultiply multiply;
        multiply.factor = 2.;
        Flip flip{"x", 1., 1.};
        Pipeline pipeline({multiply, flip}, {{"image", image}});
        auto result = pipeline.run(input, 42, true, false);
        auto replay = pipeline.replay(input, load_plan(pipeline.program(), save_plan(result.plan)));
        const auto* actual = static_cast<const float*>(result.sample.at("image").array.data());
        for (int i = 0; i < 12; ++i)
            require(actual[i] == 2.f * (i / 4 * 4 + 4 - i % 4), "CPU transform pipeline values");
        require(std::memcmp(actual, replay.sample.at("image").array.data(), 12 * sizeof(float)) == 0,
                "serialized native replay must reproduce values");
        require(values[0] == 1.f, "borrowed/copied inputs remain unchanged");
        std::cout << "Native transform pipeline and replay passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
