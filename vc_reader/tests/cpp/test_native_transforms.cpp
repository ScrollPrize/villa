#include <vc_reader/transforms.hpp>
#include <cmath>
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
        auto result = pipeline.run(input, 42, true);
        auto replay = pipeline.replay(input, load_plan(pipeline.program(), save_plan(result.plan)));
        const auto* actual = static_cast<const float*>(result.sample.at("image").array.data());
        for (int i = 0; i < 12; ++i)
            require(actual[i] == 2.f * (i / 4 * 4 + 4 - i % 4), "CPU transform pipeline values");
        require(std::memcmp(actual, replay.sample.at("image").array.data(), 12 * sizeof(float)) == 0,
                "serialized native replay must reproduce values");
        require(values[0] == 1.f, "borrowed/copied inputs remain unchanged");
        require(make_seed({}) == UINT64_C(14695981039346656037), "seed of nothing");
        require(make_seed({seed_part(std::int64_t{1}), seed_part("a")}) == UINT64_C(10903677741198114521), "stable seed");
        require(make_seed({seed_part(std::int64_t{-5})}) == UINT64_C(11174910008215009199), "negative seed part");
        require(make_seed({seed_part("1")}) != make_seed({seed_part(std::int64_t{1})}), "integers and strings differ");
        const auto defaults = default_targets();
        require(defaults.size() == 3 && defaults.at("image") == Target{} && defaults.at("mask").role == Role::Label && defaults.at("padding_mask").role == Role::Valid, "default targets");
        require(continuous(defaults.at("image")) && !continuous(defaults.at("mask")) && !continuous(defaults.at("padding_mask")), "continuous targets");

        // Configuration checks belong to the pipeline, however a transform was built.
        const std::map<std::string, Target> schema{{"image", image}};
        const auto rejected = [&](const Op& op, const char* what) {
            bool threw = false;
            try { Pipeline({op}, schema); } catch (const std::invalid_argument&) { threw = true; }
            require(threw, std::string("rejected: ") + what);
        };
        Normalize percentiles; percentiles.lower = 90; percentiles.upper = 10; rejected(percentiles, "percentile order");
        {
            // Masked Gamma without clipping: voxels below the masked minimum (the padding) must not
            // raise a negative base to a fractional power, which was NaN.
            std::vector<float> v{-10, 2, 4, 6, 20, 3}; std::vector<std::uint8_t> m{0, 1, 1, 1, 0, 1};
            Target unfilled; unfilled.refill = false; Target valid; valid.role = Role::Valid;
            Gamma g; g.gamma = .7; g.masked = true; g.clip = false;
            Sample s; s["image"] = {Array({1, 1, 2, 3}, 'f', 4), unfilled, "image"};
            std::memcpy(s.at("image").array.data(), v.data(), 6 * sizeof(float));
            Array mask({1, 1, 2, 3}, 'u', 1); std::memcpy(mask.data(), m.data(), 6); s["padding_mask"] = {std::move(mask), valid, "padding_mask"};
            const auto out = Pipeline({g}, {{"image", unfilled}, {"padding_mask", valid}}).run(s, 0, false).sample.at("image").array;
            for (std::size_t i = 0; i < 6; ++i) require(std::isfinite(out.get(i)), "masked gamma without clipping is finite below the masked minimum");
        }
        Normalize fixed; fixed.scale = 0; rejected(fixed, "fixed scale");
        BrightnessMultiply chance; chance.options.p_channel = 2; rejected(chance, "channel probability");
        Gamma gamma; gamma.p_invert = -1; rejected(gamma, "gamma inversion probability");
        OrientedBlur blur; blur.axis = {0, 0, 0}; rejected(blur, "zero axis");
        DistanceTransform distance; distance.suffix = ""; rejected(distance, "empty suffix");
        StructureTensor tensor; tensor.derivative = Derivative::Gaussian; tensor.sigma = 0; rejected(tensor, "gaussian sigma");
        CoherenceEnhancingDiffusion diffusion;
        diffusion.step_size = .2; rejected(diffusion, "unstable diffusion step");
        diffusion.step_size = .1; diffusion.alpha = 1; diffusion.num_steps = 1;
        auto diffused = Pipeline({diffusion}, schema).run(input, 0, true);
        const auto& heat = diffused.sample.at("image").array;
        double mass = 0;
        for (std::size_t i = 0; i < heat.size(); ++i) mass += heat.get(i);
        require(std::abs(mass-78) < 1e-5, "diffusion conserves intensity");
        require(std::abs(heat.get(0)-1.5) < 1e-5, "diffusion has no-flux boundaries");
        auto diffusion_replay = Pipeline({diffusion}, schema);
        const auto replayed_heat = diffusion_replay.replay(input,
            load_plan(diffusion_replay.program(), save_plan(diffused.plan)));
        require(std::memcmp(heat.data(), replayed_heat.sample.at("image").array.data(),
                            heat.size()*sizeof(float)) == 0, "diffusion native replay");
        ForegroundCrop crop; rejected(crop, "crop shape");
        ObjectDropout dropout; dropout.fill.image = std::numeric_limits<double>::infinity(); rejected(dropout, "fill value");
        HistogramMatching matching; rejected(matching, "missing reference");
        matching.reference = volume_detail::reference_volume(Array({1, 2, 2}, 'f', 4)); Pipeline({matching}, schema);
        static_cast<float*>(matching.reference.values.data())[0] = std::numeric_limits<float>::quiet_NaN(); rejected(matching, "reference values");
        auto one_of = std::make_shared<OneOf>(); one_of->transforms = {percentiles}; rejected(one_of, "inside OneOf");
        // The defaults are the native ones, whichever language builds the transform.
        require(std::holds_alternative<Uniform>(ChannelMisalignment{}.affine.degrees) && !ChannelMisalignment{}.affine.shared_scale, "channel misalignment default");
        require(SurfacePerturb{}.p == .35, "surface perturbation default");
        std::cout << "Native transform pipeline and replay passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
