#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <iomanip>
#include <iostream>
#include <random>
#include <fstream>
#include <atomic>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/Slicing.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;
using json = nlohmann::json;


class SegmentRenderer {
    std::shared_ptr<VolumePkg> vpkg_;
    std::shared_ptr<Volume> volume_;
    ChunkCache<uint8_t>* cache_;

    struct SurfaceInfo {
        std::string id;
        std::shared_ptr<QuadSurface> surface;
        int color_index;
    };

    std::map<std::string, int> segment_color_map_;

    cv::Vec3b getColormapColor(int index, int total_count) {
        int gray_value = (index * 255) / std::max(1, total_count - 1);

        cv::Mat gray(1, 1, CV_8UC1, cv::Scalar(gray_value));
        cv::Mat colored;

        cv::applyColorMap(gray, colored, cv::COLORMAP_HSV);
        return colored.at<cv::Vec3b>(0, 0);
    }

    cv::Mat generateLegend(const fs::path& output_path) {
        if (segment_color_map_.empty()) {
            std::cout << "No segments to create legend for" << std::endl;
            return cv::Mat();
        }

        // Font settings
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.7;
        int thickness = 2;
        int line_height = 30;
        int margin = 20;

        // Calculate legend dimensions
        int max_width = 0;
        for (const auto& [seg_id, _] : segment_color_map_) {
            int baseline;
            cv::Size text_size = cv::getTextSize(seg_id, font, font_scale, thickness, &baseline);
            max_width = std::max(max_width, text_size.width);
        }

        int legend_width = max_width + 2 * margin + 20;  // Extra space for color square
        int legend_height = segment_color_map_.size() * line_height + 2 * margin;

        // Create legend image with white background
        cv::Mat legend(legend_height, legend_width, CV_8UC3, cv::Scalar(255, 255, 255));

        std::string title = "Segment Colors (Alphabetical Order)";
        int title_baseline;
        cv::Size title_size = cv::getTextSize(title, font, font_scale * 0.8, thickness, &title_baseline);
        cv::putText(legend, title,
                   cv::Point((legend_width - title_size.width) / 2, margin - 5),
                   font, font_scale * 0.8, cv::Scalar(0, 0, 0), thickness - 1);

        int idx = 0;
        for (const auto& [seg_id, color_idx] : segment_color_map_) {
            cv::Vec3b color = getColormapColor(color_idx, segment_color_map_.size());

            // Convert BGR to RGB for display
            cv::Scalar text_color(color[0], color[1], color[2]);

            int y_pos = margin + (idx + 1) * line_height;

            // Draw color square
            cv::rectangle(legend,
                         cv::Point(margin - 15, y_pos - 20),
                         cv::Point(margin - 5, y_pos - 10),
                         text_color, -1);

            // Draw segment ID with its color
            cv::putText(legend, seg_id,
                       cv::Point(margin, y_pos),
                       font, font_scale, text_color, thickness);

            idx++;
        }

        fs::path legend_path = output_path.parent_path() / (output_path.stem().string() + "_legend.png");
        cv::imwrite(legend_path.string(), legend);
        std::cout << "Legend saved to: " << legend_path << std::endl;

        return legend;
    }

public:
    SegmentRenderer(const fs::path& volpkg_path, const std::string& volume_id) {
        vpkg_ = VolumePkg::New(volpkg_path.string());

        if (volume_id.empty()) {
            throw std::runtime_error("You must provide a volume id");
        }

        if (!vpkg_->hasVolume(volume_id)) {
            throw std::runtime_error("Volume not found: " + volume_id);
        }
        volume_ = vpkg_->volume(volume_id);
        std::cout << "Using volume: " << volume_id << " (" << volume_->name() << ")" << std::endl;
        auto [w, h, d] = volume_->shape();
        std::cout << "Volume dimensions: " << w << "x" << h << "x" << d << std::endl;
        cache_ = new ChunkCache<uint8_t>(1ULL * 1024ULL * 1024ULL * 1024ULL);
    }

    ~SegmentRenderer() {
        delete cache_;
    }

    cv::Mat render(const std::string& segment_id, const fs::path& output_path,
                   const std::string& source, const std::string& filter = "", int stride = 0) {

        return renderWithSpatialIndex(segment_id, output_path, source, filter, stride);
    }

private:
    cv::Mat renderWithSpatialIndex(const std::string& target_segment_id,
                                    const fs::path& output_path,
                                    const std::string& source,
                                    const std::string& filter,
                                    int stride = 0) {

        std::cout << "Rendering " << source << " for segment: " << target_segment_id << std::endl;

        segment_color_map_.clear();

        // Load target segment
        auto target_surf = vpkg_->loadSurface(target_segment_id);
        if (!target_surf) {
            throw std::runtime_error("Failed to load target segment: " + target_segment_id);
        }

        cv::Mat_<cv::Vec3f> raw_points = target_surf->rawPoints();
        cv::Vec2f stored_scale = target_surf->scale();

        int native_width = raw_points.cols / stored_scale[0];
        int native_height = raw_points.rows / stored_scale[1];

        std::cout << "Target resolution: " << native_width << "x" << native_height << std::endl;

        // Generate at native resolution
        float gen_scale = 1.0f;
        cv::Size gen_size(native_width / gen_scale, native_height / gen_scale);

        cv::Mat_<cv::Vec3f> coords;
        cv::Vec3f center = target_surf->pointer();
        cv::Vec3f offset = {
            -(float)(gen_size.width / 2),
            -(float)(gen_size.height / 2),
            0
        };

        target_surf->gen(&coords, nullptr, gen_size, center, gen_scale, offset);

        // Load surface IDs based on source with sorted color assignment
        std::vector<SurfaceInfo> surfaces = loadSurfaces(target_segment_id, target_surf,
                                                         source, filter);

        std::cout << "Loaded " << surfaces.size() << " surfaces" << std::endl;

        if (surfaces.empty() && source != "sequence") {
            std::cerr << "Warning: No surfaces found for source: " << source << std::endl;
            // Return image with just the target in black
            cv::Mat_<cv::Vec3b> output(gen_size, cv::Vec3b(255, 255, 255));

            // Draw target in black
            for (int j = 0; j < gen_size.height; j++) {
                for (int i = 0; i < gen_size.width; i++) {
                    const cv::Vec3f& point = coords(j, i);
                    if (point[0] != -1 && !std::isnan(point[0]) &&
                        !std::isnan(point[1]) && !std::isnan(point[2])) {
                        output(j, i) = cv::Vec3b(0, 0, 0);  // Black for target
                    }
                }
            }

            cv::imwrite(output_path.string(), output);
            return output;
        }

        // Build spatial index using SurfacePatchIndex
        SurfacePatchIndex patchIndex;
        std::vector<QuadSurface*> surface_ptrs;
        for (const auto& info : surfaces) {
            surface_ptrs.push_back(info.surface.get());
        }
        patchIndex.rebuild(surface_ptrs);

        std::cout << "SurfacePatchIndex built for " << surfaces.size() << " surfaces" << std::endl;

        // Process with optional stride for performance
        if (stride <= 0) {
            stride = (gen_size.width > 2000) ? 2 : 1;
            std::cout << "Auto-selected stride: " << stride << " based on width " << gen_size.width << std::endl;
        } else {
            std::cout << "Using user-specified stride: " << stride << std::endl;
        }

        cv::Size process_size(gen_size.width / stride, gen_size.height / stride);
        cv::Mat_<cv::Vec3b> output(process_size, cv::Vec3b(255, 255, 255));  // White background

        std::cout << "Processing with stride " << stride << ": "
                  << process_size.width << "x" << process_size.height << std::endl;

        // Statistics tracking
        std::atomic<int> valid_count(0), target_only_count(0);
        std::vector<std::atomic<int>> surface_counts(surfaces.size());
        for (size_t i = 0; i < surfaces.size(); i++) {
            surface_counts[i] = 0;
        }

        float tolerance = stride * 1.5f;  // Adjust tolerance based on stride

        // Process points in parallel
        #pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < process_size.height; j++) {
            if (j % 50 == 0) {
                #pragma omp critical
                {
                    std::cout << "Processing row " << j << "/" << process_size.height << std::endl;
                }
            }

            for (int i = 0; i < process_size.width; i++) {
                int src_j = j * stride;
                int src_i = i * stride;

                const cv::Vec3f& point = coords(src_j, src_i);

                // Skip invalid points - they stay white
                if (point[0] == -1 || std::isnan(point[0]) ||
                    std::isnan(point[1]) || std::isnan(point[2])) {
                    continue;
                }

                valid_count++;

                // Direct lookup using SurfacePatchIndex
                bool found_match = false;
                int matched_idx = -1;

                auto result = patchIndex.locate(point, tolerance);
                if (result.has_value()) {
                    // Find matching surface index
                    for (size_t idx = 0; idx < surfaces.size(); idx++) {
                        if (surfaces[idx].surface.get() == result->surface) {
                            matched_idx = static_cast<int>(idx);
                            found_match = true;
                            break;
                        }
                    }
                }

                if (found_match) {
                    // Point is in an overlapping surface - use colormap
                    output(j, i) = getColormapColor(surfaces[matched_idx].color_index,
                                                   segment_color_map_.size());
                    surface_counts[matched_idx]++;
                } else {
                    // Point is in target only - use black
                    output(j, i) = cv::Vec3b(0, 0, 0);
                    target_only_count++;
                }
            }
        }

        // Auto-crop if needed
        cv::Mat final_output = autoCrop(output);

        // Save output
        cv::imwrite(output_path.string(), final_output);

        // Generate and save legend
        generateLegend(output_path);

        // Print statistics
        printStatistics(surfaces, valid_count, target_only_count, surface_counts);

        std::cout << "Saved to: " << output_path << std::endl;

        return final_output;
    }

    std::vector<SurfaceInfo> loadSurfaces(const std::string& target_id,
                                          std::shared_ptr<QuadSurface> target_surf,
                                          const std::string& source,
                                          const std::string& filter) {

        std::vector<SurfaceInfo> surfaces;
        std::vector<std::string> surface_ids = getSurfaceIds(target_id, target_surf, source);

        // Apply filter if provided
        if (!filter.empty()) {
            std::cout << "Applying filter to " << surface_ids.size() << " surfaces" << std::endl;
            surface_ids = applyFilter(surface_ids, filter);
            std::cout << "After filtering: " << surface_ids.size() << " surfaces" << std::endl;
        }

        // Sort surface IDs alphabetically
        std::sort(surface_ids.begin(), surface_ids.end());

        std::cout << "Assigning colors to " << surface_ids.size() << " surfaces in alphabetical order:" << std::endl;

        // Handle sequence source specially
        if (source == "sequence") {
            return loadSequenceSurfaces(target_id, target_surf, surface_ids);
        }

        // Load all surfaces with color indices based on sorted order
        int color_idx = 0;
        for (const auto& surf_id : surface_ids) {
            auto surf = vpkg_->loadSurface(surf_id);
            if (surf) {
                surfaces.push_back({surf_id, surf, color_idx});
                segment_color_map_[surf_id] = color_idx;
                std::cout << "  " << surf_id << " -> color index " << color_idx << std::endl;
                color_idx++;
            } else {
                std::cerr << "Failed to load surface: " << surf_id << std::endl;
            }
        }

        return surfaces;
    }

    std::vector<SurfaceInfo> loadSequenceSurfaces(const std::string& target_id,
                                                  std::shared_ptr<QuadSurface> target_surf,
                                                  const std::vector<std::string>& sorted_sequence) {
        std::vector<SurfaceInfo> surfaces;

        // Build color map from sorted sequence
        int color_idx = 0;
        for (const auto& seq_id : sorted_sequence) {
            segment_color_map_[seq_id] = color_idx++;
        }

        // Load surfaces up to and including target (but use sorted color indices)
        bool found_target = false;

        // Need to get original unsorted sequence for loading order
        std::vector<std::string> original_sequence = getSurfaceIds(target_id, target_surf, "sequence");

        for (const auto& seq_id : original_sequence) {
            auto surf = vpkg_->loadSurface(seq_id);
            if (surf && segment_color_map_.find(seq_id) != segment_color_map_.end()) {
                surfaces.push_back({seq_id, surf, segment_color_map_[seq_id]});
            }

            if (seq_id == target_id) {
                found_target = true;
                break;
            }
        }

        // Add target if not in sequence
        if (!found_target) {
            // Assign a color for target if not already assigned
            if (segment_color_map_.find(target_id) == segment_color_map_.end()) {
                segment_color_map_[target_id] = segment_color_map_.size();
            }
            surfaces.push_back({target_id, target_surf, segment_color_map_[target_id]});
        }

        return surfaces;
    }

    std::vector<std::string> getSurfaceIds(const std::string& target_id,
                                           std::shared_ptr<QuadSurface> target_surf,
                                           const std::string& source) {
        std::vector<std::string> ids;

        if (source == "overlapping") {
            auto overlapping_ids = target_surf->overlappingIds();
            ids.assign(overlapping_ids.begin(), overlapping_ids.end());
        } else if (source == "contributing" || source == "approved_patches" || source == "sequence") {
            fs::path meta_path = target_surf->path / "meta.json";
            if (fs::exists(meta_path)) {
                std::ifstream meta_file(meta_path);
                json meta_json;
                meta_file >> meta_json;

                std::string json_key;
                if (source == "contributing") {
                    json_key = "contributing_surfaces";
                } else if (source == "approved_patches") {
                    json_key = "used_approved_segments";
                } else if (source == "sequence") {
                    json_key = "surface_sequence";
                }

                if (meta_json.contains(json_key)) {
                    ids = meta_json[json_key].get<std::vector<std::string>>();
                }
            }
        }

        return ids;
    }

    std::vector<std::string> applyFilter(const std::vector<std::string>& ids,
                                         const std::string& filter_str) {
        if (filter_str.empty()) return ids;

        std::unordered_set<std::string> filter_set;
        std::stringstream ss(filter_str);
        std::string id;

        while (std::getline(ss, id, ',')) {
            // Trim whitespace
            id.erase(0, id.find_first_not_of(" \t"));
            id.erase(id.find_last_not_of(" \t") + 1);
            if (!id.empty()) {
                filter_set.insert(id);
            }
        }

        std::vector<std::string> filtered;
        for (const auto& surf_id : ids) {
            if (filter_set.find(surf_id) != filter_set.end()) {
                filtered.push_back(surf_id);
            }
        }

        return filtered;
    }

    cv::Mat autoCrop(const cv::Mat_<cv::Vec3b>& image) {
        // Find bounding box of non-white pixels
        int min_x = image.cols, max_x = -1;
        int min_y = image.rows, max_y = -1;

        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                const cv::Vec3b& pixel = image(y, x);
                if (pixel != cv::Vec3b(255, 255, 255)) {
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                    min_y = std::min(min_y, y);
                    max_y = std::max(max_y, y);
                }
            }
        }

        // If no non-white pixels found, return original
        if (max_x < min_x || max_y < min_y) {
            return image;
        }

        // Add small margin
        int margin = 5;
        min_x = std::max(0, min_x - margin);
        min_y = std::max(0, min_y - margin);
        max_x = std::min(image.cols - 1, max_x + margin);
        max_y = std::min(image.rows - 1, max_y + margin);

        cv::Rect crop_rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
        return image(crop_rect).clone();
    }

    void printStatistics(const std::vector<SurfaceInfo>& surfaces,
                        int valid_count, int target_only_count,
                        const std::vector<std::atomic<int>>& surface_counts) {

        std::cout << "\n=== Rendering Statistics ===" << std::endl;
        std::cout << "Valid points processed: " << valid_count << std::endl;
        std::cout << "Points in target only (black): " << target_only_count
                  << " (" << (100.0 * target_only_count / std::max(1, valid_count)) << "%)" << std::endl;

        int surfaces_hit = 0;
        int total_overlap_points = 0;

        // Print in alphabetical order (using segment_color_map_ which is a std::map)
        for (const auto& [seg_id, color_idx] : segment_color_map_) {
            // Find the surface in the surfaces vector
            for (size_t i = 0; i < surfaces.size(); i++) {
                if (surfaces[i].id == seg_id) {
                    int count = surface_counts[i];
                    if (count > 0) {
                        std::cout << "Surface " << seg_id << ": " << count << " points (color index "
                                  << color_idx << ")" << std::endl;
                        surfaces_hit++;
                        total_overlap_points += count;
                    }
                    break;
                }
            }
        }

        std::cout << "Total overlap points (colored): " << total_overlap_points
                  << " (" << (100.0 * total_overlap_points / std::max(1, valid_count)) << "%)" << std::endl;
        std::cout << "Surfaces with matches: " << surfaces_hit << "/" << segment_color_map_.size() << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
        po::options_description desc("Segment Renderer Options");
        desc.add_options()
            ("help,h", "Show help message")
            ("volpkg-path,v", po::value<std::string>()->required(),
                "Path to volume package")
            ("volume-id", po::value<std::string>()->required(),
                "ID of volume to use")
            ("segment-id,s", po::value<std::string>()->required(),
                "ID of segment to render")
            ("overlap-source", po::value<std::string>()->required(),
                "Source for overlaps (overlapping|contributing|sequence|approved_patches)")
            ("output,o", po::value<std::string>()->required(),
                "Output PNG file path")
            ("stride", po::value<int>()->default_value(0),
                "Stride for point sampling (1=full resolution, 2=half, etc.). 0=auto-select based on size")
            ("filter", po::value<std::string>()->default_value(""),
                "Comma-separated list of surface IDs to include (works with any source)")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << "Segment Renderer - Render segment overlaps and visualizations\n\n";
            std::cout << desc << std::endl;
            std::cout << "\nColor scheme:\n";
            std::cout << "  - White: Background/invalid points\n";
            std::cout << "  - Black: Target segment only\n";
            std::cout << "  - Colors: Overlapping surfaces (HSV colormap, assigned alphabetically)\n";
            std::cout << "  - Legend: Automatically generated as <output>_legend.png\n\n";
            std::cout << "Examples:\n";
            std::cout << "  # Basic rendering with auto-selected stride:\n";
            std::cout << "  " << argv[0] << " --volpkg-path /path/to/vol.volpkg --volume-id vol123 \\\n";
            std::cout << "                  --segment-id seg456 --overlap-source overlapping \\\n";
            std::cout << "                  --output output.png\n\n";

            std::cout << "  # Full resolution rendering (stride=1):\n";
            std::cout << "  " << argv[0] << " -v /path/to/vol.volpkg --volume-id vol123 -s seg456 \\\n";
            std::cout << "                  --overlap-source approved_patches -o output.png --stride 1\n\n";

            std::cout << "  # Fast preview with stride=4:\n";
            std::cout << "  " << argv[0] << " -v /path/to/vol.volpkg --volume-id vol123 -s seg456 \\\n";
            std::cout << "                  --overlap-source contributing -o output.png --stride 4\n\n";

            std::cout << "  # Filter to specific surfaces:\n";
            std::cout << "  " << argv[0] << " -v /path/to/vol.volpkg --volume-id vol123 -s seg456 \\\n";
            std::cout << "                  --overlap-source contributing -o output.png \\\n";
            std::cout << "                  --filter \"surface1,surface2,surface3\"\n";
            return EXIT_SUCCESS;
        }

        po::notify(vm);

        fs::path volpkg_path = vm["volpkg-path"].as<std::string>();
        std::string volume_id = vm["volume-id"].as<std::string>();
        std::string segment_id = vm["segment-id"].as<std::string>();
        std::string overlap_source = vm["overlap-source"].as<std::string>();
        fs::path output_path = vm["output"].as<std::string>();
        int stride = vm["stride"].as<int>();
        std::string filter = vm["filter"].as<std::string>();

        std::set<std::string> valid_sources = {
            "overlapping", "contributing", "sequence", "approved_patches"
        };

        if (valid_sources.find(overlap_source) == valid_sources.end()) {
            std::cerr << "Error: Invalid overlap source '" << overlap_source << "'\n";
            std::cerr << "Must be one of: overlapping, contributing, sequence, approved_patches\n";
            return EXIT_FAILURE;
        }

        if (stride < 0) {
            std::cerr << "Error: Stride must be >= 0 (0 for auto-select)\n";
            return EXIT_FAILURE;
        }

        SegmentRenderer renderer(volpkg_path, volume_id);
        renderer.render(segment_id, output_path, overlap_source, filter, stride);

        return EXIT_SUCCESS;

    } catch (const po::error& e) {
        std::cerr << "Command line error: " << e.what() << "\n";
        std::cerr << "Use --help for usage information\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}