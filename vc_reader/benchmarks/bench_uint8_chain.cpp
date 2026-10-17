// Experimental storage comparison; deliberately not a public augmentation API.
// Reuses native median, permutation, and intensity kernels. See README.md.
#include "pipeline.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace vcr::augment;
using Clock = std::chrono::steady_clock;

static std::vector<Op> permutations() {
    return {Rotate90{.planes = {"yx"}, .turns = {{1}}}, Transpose{.order = "xyz"}, Flip{.axes = "z", .p_axis = 1}};
}
static std::vector<Op> intensities() {
    return {BrightnessMultiply{.factor = 1.1}, BrightnessAdd{.shift = .05}, Gamma{.gamma = .8}};
}
static Sample sample(Array a, Target target) {
    Sample s; s.emplace("image", Field{std::move(a), target, "image"}); return s;
}
static Array run(const std::vector<std::uint8_t>& raw, std::size_t edge, int median,
                 int stages, int mode, double* clipped = nullptr) {
    // mode 0: ordinary float32; 1: fixed [0,1] uint8; 2: uint8 + float palette.
    Target target; if (mode) target.role = Role::Label; // permutations only; no interpolation
    Array a({1, edge, edge, edge}, mode ? 'u' : 'f', mode ? 1 : 4);
    if (mode) std::memcpy(a.data(), raw.data(), raw.size());
    else {
        auto* p = static_cast<float*>(a.data());
        for (std::size_t i = 0; i < raw.size(); ++i) p[i] = raw[i] / 255.f;
    }
    auto ops = permutations();
    if (!mode) {
        if (median) ops.insert(ops.begin(), MedianFilter{.size = double(median)});
        auto intensity = intensities(); ops.insert(ops.end(), intensity.begin(), intensity.begin() + stages);
        auto result = Pipeline(ops, {{"image", target}}, false).run(sample(std::move(a), target), 123, false, false);
        return std::move(result.sample.at("image").array);
    }
    if (median) {
        auto* p = static_cast<std::uint8_t*>(a.data());
        std::vector<std::uint8_t> v(p, p + raw.size());
        channel_apply(MedianFilter{}, v, Grid(a.spatial()), ChannelPlan{"image", 0, {double(median)}});
        std::copy(v.begin(), v.end(), p);
    }
    auto result = Pipeline(ops, {{"image", target}}, false).run(sample(std::move(a), target), 123, false, false);
    a = std::move(result.sample.at("image").array);
    auto* codes = static_cast<std::uint8_t*>(a.data());
    std::array<float, 256> palette;
    for (std::size_t i = 0; i < palette.size(); ++i) palette[i] = i / 255.f;
    auto intensity = intensities();
    for (int stage = 0; stage < stages; ++stage) {
        // Gamma uses the actual occupied range. Unoccupied palette entries
        // must not change its min/max. These transforms are monotone.
        auto bounds = std::minmax_element(codes, codes + raw.size());
        const auto low = palette[*bounds.first], high = palette[*bounds.second];
        Array lut({1, 16, 16});
        auto* data = static_cast<float*>(lut.data());
        for (std::size_t i = 0; i < palette.size(); ++i) data[i] = std::clamp(palette[i], low, high);
        auto mapped = Pipeline({intensity[stage]}, {{"image", Target{}}}, false).run(sample(std::move(lut), Target{}), 123, false, false);
        auto* values = static_cast<const float*>(mapped.sample.at("image").array.data());
        if (mode == 2) std::copy_n(values, palette.size(), palette.begin());
        else {
            std::array<std::uint8_t, 256> table;
            for (std::size_t i = 0; i < table.size(); ++i)
                table[i] = std::uint8_t(std::clamp(std::floor(double(values[i]) * 255 + .5), 0., 255.));
            for (std::size_t i = 0; i < raw.size(); ++i) {
                if (clipped) *clipped += values[codes[i]] < 0 || values[codes[i]] > 1;
                codes[i] = table[codes[i]];
            }
        }
    }
    Array output(a.shape);
    auto* dest = static_cast<float*>(output.data());
    for (std::size_t i = 0; i < raw.size(); ++i) dest[i] = palette[codes[i]];
    return output;
}

static void stats(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto percentile = [&](double p) {
        double index = p * (values.size() - 1); auto low = std::size_t(index);
        return std::lerp(values[low], values[std::min(low + 1, values.size() - 1)], index - low);
    };
    std::cout << "{\"mean_ms\":" << mean(values) << ",\"p50_ms\":" << percentile(.5)
              << ",\"p95_ms\":" << percentile(.95) << ",\"min_ms\":" << values.front()
              << ",\"max_ms\":" << values.back() << '}';
}
#include "palette_extended.hpp"

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: bench_uint8_chain EDGE REPS MEDIAN_SIZE RAW_CROP...\n"; return 2; }
    const auto edge = std::stoul(argv[1]); const auto reps = std::stoi(argv[2]), median = std::stoi(argv[3]);
    require(edge > 0 && reps > 0 && median > 0 && median <= 63, "invalid workload");
    std::vector<std::vector<std::uint8_t>> crops;
    const bool extended = std::string(argv[4]) == "--extended";
    for (int arg = extended ? 5 : 4; arg < argc; ++arg) {
        std::vector<std::uint8_t> raw(edge * edge * edge);
        std::ifstream input(argv[arg], std::ios::binary);
        require(bool(input.read(reinterpret_cast<char*>(raw.data()), raw.size())), "short raw crop");
        crops.push_back(std::move(raw));
    }
    require(!crops.empty(), "at least one crop is required");
    if (extended) { extended_benchmark(crops, edge, reps); return 0; }
    std::cout << std::setprecision(10) << "{\"edge\":" << edge << ",\"reps\":" << reps
              << ",\"crops\":" << crops.size() << ",\"median_size\":" << median << ",\"results\":[";
    bool first = true;
    for (int stage_count = -1; stage_count <= 3; ++stage_count) {
        int window = stage_count < 0 ? 0 : median, stages = std::max(0, stage_count);
        std::array<std::vector<double>, 3> timing;
        std::array<double, 3> max_error{}, squared{}, bias{}, changed{}, clipped{};
        // Untimed warmup/correctness pass. Prefix cases expose accumulated error.
        for (const auto& raw : crops) {
            auto reference = run(raw, edge, window, stages, 0);
            auto* expected = static_cast<const float*>(reference.data());
            for (int mode = 1; mode < 3; ++mode) {
                auto output = run(raw, edge, window, stages, mode, &clipped[mode]);
                auto* actual = static_cast<const float*>(output.data());
                for (std::size_t i = 0; i < raw.size(); ++i) {
                    const double error = double(actual[i]) - expected[i];
                    max_error[mode] = std::max(max_error[mode], std::abs(error));
                    squared[mode] += error * error; bias[mode] += error; changed[mode] += error != 0;
                }
            }
        }
        require(max_error[2] == 0, "palette result differs from native float32");
        if (!stages) require(max_error[1] == 0, "uint8 permutation/median changed values");
        constexpr int orders[6][3] = {{0, 1, 2}, {2, 1, 0}, {1, 2, 0}, {0, 2, 1}, {2, 0, 1}, {1, 0, 2}};
        for (int rep = 0; rep < reps; ++rep) for (const auto& raw : crops)
            for (int slot = 0; slot < 3; ++slot) {
                const int mode = orders[rep % 6][slot];
                auto start = Clock::now(); auto output = run(raw, edge, window, stages, mode);
                timing[mode].push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
                require(std::isfinite(output.get(0)), "nonfinite output"); // keep result observable, untimed
            }
        for (int mode = 0; mode < 3; ++mode) {
            if (!first) std::cout << ','; first = false;
            const double count = double(edge * edge * edge) * crops.size();
            std::cout << "{\"median\":" << window << ",\"intensity_stages\":" << stages << ",\"mode\":\""
                      << (mode == 0 ? "float32" : mode == 1 ? "uint8" : "palette") << "\",\"latency\":";
            stats(timing[mode]);
            std::cout << ",\"max_absolute\":" << max_error[mode] << ",\"rmse\":" << std::sqrt(squared[mode] / count)
                      << ",\"bias\":" << bias[mode] / count << ",\"changed_fraction\":" << changed[mode] / count
                      << ",\"clipped_stage_fraction\":" << clipped[mode] / (count * std::max(1, stages)) << '}';
        }
        std::cerr << "completed intensity stages=" << stage_count << '\n';
    }
    std::cout << "]}\n";
}
