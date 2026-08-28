#include "patch_merger.hpp"

#include <charconv>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using vc_spiral::patch_merger::MergeOptions;

void usage(std::ostream& stream)
{
    stream
        << "Usage: merge_overlapping_patches PATCH_DIR OUTPUT_DIR [options]\n"
        << "\n"
        << "Direct-overlap TIFXYZ merger. Defaults:\n"
        << "  --tolerance 2\n"
        << "  --dense-spacing 1\n"
        << "  --erode-cells 1\n"
        << "  --output-step 20\n"
        << "  --thinning-spacing 2\n"
        << "  --max-correspondences 4096\n"
        << "  --uv-inlier-tolerance 3\n"
        << "  --min-inliers 32\n"
        << "  --min-major-spread 20\n"
        << "  --min-minor-spread 5\n"
        << "  --max-refit-rms 2\n"
        << "  --containment-threshold 0.9\n"
        << "  --ransac-confidence 0.999\n"
        << "  --ransac-max-hypotheses 512\n"
        << "  --threads 0\n"
        << "  --allow-reflection | --no-reflection\n"
        << "  --benchmark   Print the timing report to stderr as well as stdout.\n";
}

template <typename T>
T parse_number(std::string_view text, const std::string& option)
{
    T value{};
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument("invalid value for " + option + ": " + std::string(text));
    }
    return value;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        usage(std::cout);
        return 0;
    }
    if (argc < 3) {
        usage(std::cerr);
        return 2;
    }
    MergeOptions options;
    bool benchmark = false;
    try {
        for (int index = 3; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--allow-reflection") {
                options.allow_reflection = true;
                continue;
            }
            if (option == "--no-reflection") {
                options.allow_reflection = false;
                continue;
            }
            if (option == "--benchmark") {
                benchmark = true;
                options.progress = true;
                continue;
            }
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + option);
            }
            const std::string_view value = argv[++index];
            if (option == "--tolerance") options.tolerance = parse_number<double>(value, option);
            else if (option == "--dense-spacing") options.dense_spacing = parse_number<double>(value, option);
            else if (option == "--erode-cells") options.erode_cells = parse_number<int>(value, option);
            else if (option == "--output-step") options.output_step = parse_number<double>(value, option);
            else if (option == "--thinning-spacing") options.thinning_spacing = parse_number<double>(value, option);
            else if (option == "--max-correspondences") options.max_correspondences = parse_number<std::size_t>(value, option);
            else if (option == "--uv-inlier-tolerance") options.uv_inlier_tolerance = parse_number<double>(value, option);
            else if (option == "--min-inliers") options.min_inliers = parse_number<std::size_t>(value, option);
            else if (option == "--min-major-spread") options.min_major_spread = parse_number<double>(value, option);
            else if (option == "--min-minor-spread") options.min_minor_spread = parse_number<double>(value, option);
            else if (option == "--max-refit-rms" || option == "--max-rms") options.max_refit_rms = parse_number<double>(value, option);
            else if (option == "--containment-threshold") options.containment_threshold = parse_number<double>(value, option);
            else if (option == "--ransac-confidence") options.ransac_confidence = parse_number<double>(value, option);
            else if (option == "--ransac-max-hypotheses") options.ransac_max_hypotheses = parse_number<std::size_t>(value, option);
            else if (option == "--threads") options.threads = parse_number<int>(value, option);
            else throw std::invalid_argument("unknown option: " + option);
        }
        const auto report = vc_spiral::patch_merger::merge_patch_directory(
            argv[1], argv[2], options);
        const std::string document = vc_spiral::patch_merger::report_json(report);
        std::cout << document << '\n';
        if (benchmark) std::cerr << "benchmark report:\n" << document << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "merge_overlapping_patches: " << error.what() << '\n';
        return 1;
    }
}
