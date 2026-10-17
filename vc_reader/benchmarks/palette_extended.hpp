// Included by bench_uint8_chain.cpp; experimental consumers of native kernels.
// Statistics weight actual voxels, not equally weighted palette entries.
struct CompactImage {
    Array codes, expanded;
    std::vector<float> palette;
    std::array<std::size_t, 256> counts{};
    bool counts_valid = false, floating = false;
    int promotions = 0;
    double half_bin = 0;

    CompactImage(const std::vector<std::uint8_t>& raw, std::size_t edge)
        : codes(Array::uninitialized({1, edge, edge, edge}, 'u', 1)), palette(256) {
        std::memcpy(codes.data(), raw.data(), raw.size());
        for (std::size_t i = 0; i < 256; ++i) palette[i] = i / 255.f;
    }
    void histogram() {
        if (counts_valid) return;
        counts.fill(0);
        const auto* p = static_cast<const std::uint8_t*>(codes.data());
        for (std::size_t i = 0; i < codes.size(); ++i) ++counts[p[i]];
        counts_valid = true;
    }
    std::pair<double, double> bounds(const std::vector<float>& values) {
        histogram(); return palette_bounds(values, counts);
    }
    std::pair<double, double> statistics(const std::vector<float>& values) {
        histogram(); return palette_statistics(values, counts, codes.size());
    }
    Array decode() {
        if (floating) return std::move(expanded);
        auto output = Array::uninitialized(codes.shape, 'f', 4);
        const auto* p = static_cast<const std::uint8_t*>(codes.data());
        auto* q = static_cast<float*>(output.data());
        for (std::size_t i = 0; i < codes.size(); ++i) q[i] = palette[p[i]];
        return output;
    }
    void float_step(const Op& op, const Parameters& parameters) {
        auto s = sample(std::move(expanded), Target{});
        execute(op, s, parameters); expanded = std::move(s.at("image").array);
    }
    void promote(const Op& op, const Parameters& parameters) {
        expanded = decode(); floating = true; ++promotions; float_step(op, parameters);
    }
    void step(const Op& op, const Parameters& parameters) {
        if (floating) { float_step(op, parameters); return; }
        std::visit([&](const auto& t) {
            using T = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<T, Contrast> || std::is_same_v<T, Invert> || std::is_same_v<T, Gamma>) {
                for (const auto& channel : parameters.channels) {
                    require(channel.channel == 0, "prototype is single-channel");
                    if constexpr (std::is_same_v<T, Gamma>) {
                        histogram(); const auto b = bounds(palette);
                        for (std::size_t i = 0; i < 256; ++i) if (!counts[i]) palette[i] = float(b.first);
                        gamma_values(palette, channel,
                            [&](const auto& values) { return statistics(values); },
                            [&](const auto& values) { return bounds(values); });
                    } else {
                        const auto b = bounds(palette); const double m = statistics(palette).first;
                        for (auto& value : palette)
                            value = float(pointwise_intensity(t, value, channel, m, b.first, b.second));
                    }
                }
            } else if constexpr (std::is_same_v<T, CropPad> || std::is_same_v<T, ThickSlice>) {
                Target target; target.role = Role::Label;
                if constexpr (std::is_same_v<T, CropPad>) {
                    bool padding = false;
                    for (std::size_t a = 0; a < parameters.shape.size(); ++a)
                        padding |= parameters.offset[a] < 0 || parameters.offset[a] + parameters.shape[a] > codes.shape[a + 1];
                    if (padding) {
                        auto fill = std::find(palette.begin(), palette.end(), 0.f);
                        if (fill == palette.end()) {
                            histogram(); auto unused = std::find(counts.begin(), counts.end(), 0);
                            if (unused == counts.end()) { promote(op, parameters); return; }
                            fill = palette.begin() + (unused - counts.begin()); *fill = 0.f;
                        }
                        target.fill = fill - palette.begin();
                    }
                }
                auto s = sample(std::move(codes), target);
                if constexpr (std::is_same_v<T, ThickSlice>) {
                    auto byte_transform = t; byte_transform.labels = true;
                    apply(byte_transform, s, parameters);
                } else apply(t, s, parameters);
                codes = std::move(s.at("image").array); counts_valid = false;
            } else if constexpr (std::is_same_v<T, SliceIllumination>) {
                for (const auto& channel : parameters.channels) {
                    require(channel.channel == 0 && channel.values[0] == 0, "prototype uses one channel, z slices");
                    histogram(); const auto b = bounds(palette);
                    const auto slices = codes.shape[1], plane = codes.size() / slices;
                    std::vector<float> tables(slices * 256);
                    for (std::size_t z = 0; z < slices; ++z) for (std::size_t i = 0; i < 256; ++i)
                        tables[z * 256 + i] = counts[i] ? palette[i] : float(b.first);
                    // Apply the existing illumination kernel to 256 entries
                    // per slice. The same sampled curve is used by both paths.
                    channel_apply(t, tables, Grid({slices, 256}), channel);
                    const auto extrema = std::minmax_element(tables.begin(), tables.end());
                    const double low = *extrema.first, high = *extrema.second, range = high - low;
                    half_bin = range / 510;
                    std::vector<std::uint8_t> lookup(tables.size());
                    for (std::size_t i = 0; i < tables.size(); ++i)
                        lookup[i] = range > 0 ? std::uint8_t(std::clamp(std::floor((tables[i] - low) / range * 255 + .5), 0., 255.)) : 0;
                    auto* p = static_cast<std::uint8_t*>(codes.data());
                    for (std::size_t z = 0; z < slices; ++z)
                        std::transform(p + z * plane, p + (z + 1) * plane, p + z * plane,
                                       [&](auto code) { return lookup[z * 256 + code]; });
                    for (std::size_t i = 0; i < 256; ++i) palette[i] = float(std::lerp(low, high, i / 255.));
                    counts_valid = false;
                }
            } else throw std::invalid_argument("unsupported compact experiment transform");
        }, op);
    }
};

static Array expanded_input(const std::vector<std::uint8_t>& raw, std::size_t edge) {
    auto a = Array::uninitialized({1, edge, edge, edge}, 'f', 4);
    auto* p = static_cast<float*>(a.data());
    for (std::size_t i = 0; i < raw.size(); ++i) p[i] = raw[i] / 255.f;
    return a;
}
struct ExtendedOutput { Array image; int promotions = 0; double half_bin = 0; };
static ExtendedOutput extended_run(const std::vector<std::uint8_t>& raw, std::size_t edge,
                                   const Pipeline& pipeline, const Plan& plan, bool compact) {
    if (!compact) {
        auto result = pipeline.replay(sample(expanded_input(raw, edge), Target{}), plan);
        return {std::move(result.sample.at("image").array)};
    }
    CompactImage state(raw, edge);
    for (const auto& step : plan.steps) state.step(resolve(*plan.program, step.path), step.parameters);
    return {state.decode(), state.promotions, state.half_bin};
}

static void extended_benchmark(const std::vector<std::vector<std::uint8_t>>& crops, std::size_t edge, int reps) {
    require(edge > 16, "extended workload needs edge > 16");
    const Op contrast = Contrast{.factor = 1.2};
    const Op invert = Invert{};
    const Op gamma = Gamma{.gamma = .8, .p_retain_stats = 1};
    const Op gamma_inverted = Gamma{.gamma = 1.3, .p_invert = 1, .p_retain_stats = 1};
    const Op crop = CropPad{.shape = {edge - 8, edge - 8, edge - 8}};
    const Op pad = CropPad{.shape = {edge + 8, edge + 8, edge + 8}};
    const Op restore = CropPad{.shape = {edge, edge, edge}};
    const Op thick = ThickSlice{.scale = .5, .axes = "z"};
    const Op illumination = SliceIllumination{.defects = Uniform{2, 5}, .width = Uniform{25, 50},
        .reduction = Uniform{.3, 1.5}, .base_p = Uniform{.2, .4}, .base_reduction = Uniform{.5, .9}};
    struct Case { std::string name; std::vector<Op> ops; bool approximate = false; };
    const std::vector<Case> cases = {
        {"Contrast", {contrast}}, {"Invert", {invert}}, {"GammaRetain", {gamma}},
        {"GammaInvertRetain", {gamma_inverted}}, {"CropPad_crop", {crop}}, {"CropPad_pad", {pad}},
        {"ThickSlice", {thick}}, {"SliceIllumination", {illumination}, true},
        {"StatisticsChain", {contrast, gamma_inverted, gamma, invert}},
        {"GeometryStatisticsChain", {crop, restore, thick, contrast, gamma_inverted, gamma, invert}},
        {"FullChain_slice_first", {crop, restore, thick, illumination, contrast, gamma_inverted, gamma, invert}, true},
        {"FullChain_slice_last", {crop, restore, thick, contrast, gamma_inverted, gamma, invert, illumination}, true},
        {"PadAfterStatistics", {contrast, gamma, invert, pad}}
    };
    std::cout << std::setprecision(10) << "{\"suite\":\"extended\",\"edge\":" << edge << ",\"reps\":" << reps
              << ",\"crops\":" << crops.size() << ",\"results\":[";
    bool first = true;
    for (const auto& test : cases) {
        Pipeline pipeline(test.ops, {{"image", Target{}}}, false);
        std::vector<Plan> plans;
        std::array<std::vector<double>, 2> timing;
        double maximum = 0, squared = 0, bias = 0, changed = 0, count = 0, slice_error = 0, half_bin = 0;
        int promotions = 0;
        bool bitwise_equal = true;
        for (std::size_t c = 0; c < crops.size(); ++c) {
            auto reference = pipeline.run(sample(expanded_input(crops[c], edge), Target{}), 123 + c, true, false);
            plans.push_back(reference.plan);
            auto result = extended_run(crops[c], edge, pipeline, plans.back(), true);
            auto& expected = reference.sample.at("image").array;
            require(result.image.shape == expected.shape, "shape mismatch");
            const auto* a = static_cast<const float*>(expected.data());
            const auto* b = static_cast<const float*>(result.image.data());
            bitwise_equal &= std::memcmp(a, b, expected.size() * sizeof(float)) == 0;
            double crop_maximum = 0;
            const auto plane = expected.size() / expected.shape[1];
            for (std::size_t z = 0; z < expected.shape[1]; ++z) {
                double slice_sum = 0;
                for (std::size_t i = z * plane; i < (z + 1) * plane; ++i) {
                    require(std::isfinite(b[i]), "nonfinite compact output");
                    const double error = double(b[i]) - a[i];
                    crop_maximum = std::max(crop_maximum, std::abs(error));
                    maximum = std::max(maximum, std::abs(error)); squared += error * error;
                    bias += error; changed += error != 0; slice_sum += error;
                }
                slice_error = std::max(slice_error, std::abs(slice_sum / plane));
            }
            count += expected.size(); promotions += result.promotions;
            half_bin = std::max(half_bin, result.half_bin);
            if (test.name == "SliceIllumination" || test.name == "FullChain_slice_last")
                require(crop_maximum <= result.half_bin + 2e-6, "slice illumination exceeded half-bin bound");
        }
        if (!test.approximate) require(maximum <= 2e-6, "unexpected palette statistics error");
        for (int rep = 0; rep < reps; ++rep) for (std::size_t c = 0; c < crops.size(); ++c)
            for (int slot = 0; slot < 2; ++slot) {
                int mode = (slot + rep) % 2;
                auto start = Clock::now(); auto result = extended_run(crops[c], edge, pipeline, plans[c], mode);
                timing[mode].push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
                require(std::isfinite(result.image.get(0)), "nonfinite result");
            }
        for (int mode = 0; mode < 2; ++mode) {
            if (!first) std::cout << ','; first = false;
            std::cout << "{\"name\":\"" << test.name << "\",\"mode\":\"" << (mode ? "palette" : "float32") << "\",\"latency\":";
            stats(timing[mode]);
            std::cout << ",\"max_absolute\":" << (mode ? maximum : 0) << ",\"rmse\":" << (mode ? std::sqrt(squared / count) : 0)
                      << ",\"bias\":" << (mode ? bias / count : 0) << ",\"changed_fraction\":" << (mode ? changed / count : 0)
                      << ",\"max_slice_mean_error\":" << (mode ? slice_error : 0)
                      << ",\"half_bin\":" << (mode ? half_bin : 0) << ",\"promotions\":" << (mode ? promotions : 0)
                      << ",\"bitwise_equal\":" << ((!mode || bitwise_equal) ? "true" : "false") << '}';
        }
        std::cerr << "completed " << test.name << '\n';
    }
    std::cout << "]}\n";
}
