#include <unordered_set>
#include <algorithm>
#include <random>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <random>
#include <fstream>
#include <nlohmann/json.hpp>

#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/Slicing.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

struct SegmentData {
    cv::Mat_<cv::Vec3b> image;
    cv::Mat_<uint8_t> mask;
    cv::Vec2f offset;
    cv::Vec3b color;
};

struct AlignmentResult {
    cv::Vec2f pixel_offset;
    int num_correspondences;
    bool valid;
};

class SegmentRenderer {
private:
    std::shared_ptr<VolumePkg> vpkg_;
    std::shared_ptr<Volume> volume_;
    ChunkCache* cache_;

    cv::Vec3b getColormapColor(int index, int total_count) {
        // Map index to 0-255 range
        int gray_value = (index * 255) / std::max(1, total_count - 1);

        // Create single pixel grayscale image
        cv::Mat gray(1, 1, CV_8UC1, cv::Scalar(gray_value));
        cv::Mat colored;

        // Apply viridis colormap
        cv::applyColorMap(gray, colored, cv::COLORMAP_VIRIDIS);

        // Extract and return the color
        return colored.at<cv::Vec3b>(0, 0);
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
        std::cout << "Volume dimensions: " << volume_->sliceWidth() << "x" << volume_->sliceHeight() << "x" << volume_->numSlices() << std::endl;
        std::cout << "Available scales: " << volume_->numScales() << std::endl;
        cache_ = new ChunkCache(1ULL * 1024ULL * 1024ULL * 1024ULL);
    }

    ~SegmentRenderer() {
        delete cache_;
    }

    cv::Mat render(const std::string& segment_id, const fs::path& output_path,
                   std::string source, float opacity = 0.4, const std::string& patch_filter = "") {

        // Special handling for sequence and approved_patches sources
        if (source == "sequence") {
            return renderSequence(segment_id, output_path, opacity);
        }

        if (source == "approved_patches") {
            return renderApprovedPatches(segment_id, output_path, opacity, patch_filter);
        }

        if (source == "duplicate_check") {
            return renderDuplicateCheck(segment_id, output_path);
        }

        // Handle contributing and overlapping sources with alignment
        std::cout << "Rendering " << source << " surfaces for segment: " << segment_id << std::endl;

        // Load target segment
        auto target_meta = vpkg_->loadSurface(segment_id);
        if (!target_meta) {
            throw std::runtime_error("Failed to load target segment: " + segment_id);
        }

        QuadSurface* target_surf = target_meta->surface();

        // Get or generate mask and image for target
        auto [target_image, target_mask] = loadOrGenerateMaskedImage(target_surf, target_meta->path);

        // Get overlapping/contributing segments based on source
        std::vector<std::string> surface_ids = getOverlapIds(segment_id, target_meta, source);

        std::cout << "Found " << surface_ids.size() << " " << source << " surfaces" << std::endl;

        // Find maximum dimensions needed
        int max_width = target_image.cols;
        int max_height = target_image.rows;

        // Check all surfaces to find max size after alignment
        std::vector<SegmentData> surfaces;
        int total_surfaces = surface_ids.size() + 1; // +1 for target

        // Process each surface
        for (size_t idx = 0; idx < surface_ids.size(); idx++) {
            const std::string& surface_id = surface_ids[idx];
            std::cout << "Processing " << source << " surface [" << idx << "]: " << surface_id << std::endl;

            auto surface_meta = vpkg_->loadSurface(surface_id);
            if (!surface_meta) {
                std::cerr << "Failed to load surface: " << surface_id << std::endl;
                continue;
            }

            QuadSurface* surface = surface_meta->surface();

            // Find alignment between target and this surface
            AlignmentResult alignment = findAlignment(target_surf, surface);

            if (!alignment.valid) {
                std::cerr << "Failed to align: " << surface_id << std::endl;
                continue;
            }

            std::cout << "Alignment found with " << alignment.num_correspondences
                     << " points, offset: " << alignment.pixel_offset << std::endl;

            // Get or generate mask and image for surface
            auto [surface_image, surface_mask] = loadOrGenerateMaskedImage(surface, surface_meta->path);

            // Get colormap color based on position (idx+1 because 0 is for target)
            cv::Vec3b color = getColormapColor(idx + 1, total_surfaces);

            surfaces.push_back({
                surface_image,
                surface_mask,
                alignment.pixel_offset,
                color
            });

            // Update max dimensions considering alignment
            max_width = std::max(max_width, (int)(alignment.pixel_offset[0] + surface_image.cols));
            max_height = std::max(max_height, (int)(alignment.pixel_offset[1] + surface_image.rows));
        }

        // Create output canvas centered
        cv::Mat_<cv::Vec3b> output(cv::Size(max_width, max_height), cv::Vec3b(0, 0, 0));
        cv::Mat_<uint8_t> written_mask(output.size(), (uint8_t)0);

        // Draw target segment at origin (0,0) with first colormap color
        cv::Vec3b target_color = getColormapColor(0, total_surfaces);
        for (int j = 0; j < target_image.rows; j++) {
            for (int i = 0; i < target_image.cols; i++) {
                if (target_mask(j, i)) {
                    output(j, i) = target_color;
                    written_mask(j, i) = 1;
                }
            }
        }

        // Draw contributing/overlapping surfaces with alignment offsets
        for (const auto& surface : surfaces) {
            for (int j = 0; j < surface.image.rows; j++) {
                for (int i = 0; i < surface.image.cols; i++) {
                    if (surface.mask(j, i)) {
                        int out_x = i + surface.offset[0];
                        int out_y = j + surface.offset[1];

                        if (out_x >= 0 && out_x < output.cols &&
                            out_y >= 0 && out_y < output.rows) {
                            if (!written_mask(out_y, out_x)) {
                                output(out_y, out_x) = surface.color;
                                written_mask(out_y, out_x) = 1;
                            }
                        }
                    }
                }
            }
        }

        // Auto-crop to remove black borders
        cv::Mat final_image = autoCrop(output, written_mask);

        // Save output
        cv::imwrite(output_path.string(), final_image);
        std::cout << "Saved to: " << output_path << std::endl;

        return final_image;
    }

private:

cv::Mat renderDuplicateCheck(const std::string& segment_id, const fs::path& output_path) {
    // Hardcoded tolerance - change this value as needed
    // 0 = exact matching, 1 = 2x2x2 blocks, 2 = 4x4x4 blocks, 3 = 8x8x8 blocks
    int tolerance = 3;

    std::cout << "Checking for duplicate sampled points in segment: " << segment_id << std::endl;
    std::cout << "Using tolerance level: " << tolerance << " (groups points in "
              << (1 << tolerance) << "x" << (1 << tolerance) << "x" << (1 << tolerance)
              << " voxel blocks)" << std::endl;

    // Load segment
    auto segment_meta = vpkg_->loadSurface(segment_id);
    if (!segment_meta) {
        throw std::runtime_error("Failed to load segment: " + segment_id);
    }

    QuadSurface* surf = segment_meta->surface();
    cv::Mat_<cv::Vec3f> raw_points = surf->rawPoints();

    // Create output image - white background for invalid points
    cv::Mat_<cv::Vec3b> output(raw_points.size(), cv::Vec3b(255, 255, 255));

    // Track which integer voxel coordinates we've seen
    std::unordered_set<std::string> seen_voxels;

    // Statistics
    int valid_count = 0;
    int duplicate_count = 0;
    int invalid_count = 0;

    // Create bitmask for zeroing low bits based on tolerance
    int bitmask = ~((1 << tolerance) - 1);

    std::cout << "Processing " << raw_points.cols << "x" << raw_points.rows << " points..." << std::endl;

    // Iterate through all raw points (no interpolation)
    for (int j = 0; j < raw_points.rows; j++) {
        if (j % 100 == 0) {
            std::cout << "Processing row " << j << "/" << raw_points.rows << std::endl;
        }

        for (int i = 0; i < raw_points.cols; i++) {
            const cv::Vec3f& point = raw_points(j, i);

            // Check if point is invalid (-1,-1,-1 or NaN)
            if (point[0] == -1 || std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2])) {
                // Invalid points stay white (already set)
                invalid_count++;
                continue;
            }

            // Round coordinates to get integer voxel indices
            int voxel_x = std::round(point[0]);
            int voxel_y = std::round(point[1]);
            int voxel_z = std::round(point[2]);

            // Apply tolerance by zeroing low bits
            voxel_x = voxel_x & bitmask;
            voxel_y = voxel_y & bitmask;
            voxel_z = voxel_z & bitmask;

            // Create unique key for this voxel
            std::string voxel_key = std::to_string(voxel_x) + "," +
                                   std::to_string(voxel_y) + "," +
                                   std::to_string(voxel_z);

            // Check if we've seen this voxel before
            if (seen_voxels.find(voxel_key) != seen_voxels.end()) {
                // Duplicate found - mark as red
                output(j, i) = cv::Vec3b(0, 0, 255); // BGR format, so (0,0,255) is red
                duplicate_count++;
            } else {
                // First time seeing this voxel - mark as black (valid)
                output(j, i) = cv::Vec3b(0, 0, 0);
                seen_voxels.insert(voxel_key);
                valid_count++;
            }
        }
    }

    // Print statistics
    std::cout << "\n=== Duplicate Check Statistics ===" << std::endl;
    std::cout << "Total points: " << (raw_points.cols * raw_points.rows) << std::endl;
    std::cout << "Valid unique points: " << valid_count << std::endl;
    std::cout << "Duplicate points: " << duplicate_count;
    if (duplicate_count > 0) {
        float duplicate_percentage = (100.0f * duplicate_count) / (valid_count + duplicate_count);
        std::cout << " (" << std::fixed << std::setprecision(2) << duplicate_percentage << "% of valid points)";
    }
    std::cout << std::endl;
    std::cout << "Invalid points: " << invalid_count << std::endl;
    std::cout << "Unique voxels sampled: " << seen_voxels.size() << std::endl;

    // Optional: Create a heatmap showing duplicate density
    if (duplicate_count > 0) {
        std::cout << "\nDuplicates detected! Creating additional heatmap..." << std::endl;

        // Count duplicates per voxel
        std::unordered_map<std::string, int> voxel_counts;
        for (int j = 0; j < raw_points.rows; j++) {
            for (int i = 0; i < raw_points.cols; i++) {
                const cv::Vec3f& point = raw_points(j, i);
                if (point[0] == -1 || std::isnan(point[0])) continue;

                int voxel_x = std::round(point[0]);
                int voxel_y = std::round(point[1]);
                int voxel_z = std::round(point[2]);

                // Apply same tolerance by zeroing low bits
                voxel_x = voxel_x & bitmask;
                voxel_y = voxel_y & bitmask;
                voxel_z = voxel_z & bitmask;

                std::string voxel_key = std::to_string(voxel_x) + "," +
                                       std::to_string(voxel_y) + "," +
                                       std::to_string(voxel_z);
                voxel_counts[voxel_key]++;
            }
        }

        // Find max count for normalization
        int max_count = 0;
        for (const auto& [key, count] : voxel_counts) {
            if (count > max_count) max_count = count;
        }

        // Create heatmap
        cv::Mat_<cv::Vec3b> heatmap(raw_points.size(), cv::Vec3b(255, 255, 255));
        for (int j = 0; j < raw_points.rows; j++) {
            for (int i = 0; i < raw_points.cols; i++) {
                const cv::Vec3f& point = raw_points(j, i);
                if (point[0] == -1 || std::isnan(point[0])) continue;

                int voxel_x = std::round(point[0]);
                int voxel_y = std::round(point[1]);
                int voxel_z = std::round(point[2]);

                // Apply same tolerance by zeroing low bits
                voxel_x = voxel_x & bitmask;
                voxel_y = voxel_y & bitmask;
                voxel_z = voxel_z & bitmask;

                std::string voxel_key = std::to_string(voxel_x) + "," +
                                       std::to_string(voxel_y) + "," +
                                       std::to_string(voxel_z);

                int count = voxel_counts[voxel_key];
                if (count == 1) {
                    heatmap(j, i) = cv::Vec3b(0, 0, 0); // Black for unique
                } else {
                    // Color based on duplicate count - from yellow to red
                    float intensity = std::min(1.0f, (float)(count - 1) / (max_count - 1));
                    int green = 255 * (1.0f - intensity);
                    heatmap(j, i) = cv::Vec3b(0, green, 255); // BGR
                }
            }
        }

        // Save heatmap
        fs::path heatmap_path = output_path.parent_path() / (output_path.stem().string() + "_heatmap.png");
        cv::imwrite(heatmap_path.string(), heatmap);
        std::cout << "Heatmap saved to: " << heatmap_path << std::endl;
        std::cout << "Max duplicates for single voxel: " << max_count << std::endl;
    }

    // Save main output
    cv::imwrite(output_path.string(), output);
    std::cout << "Duplicate check visualization saved to: " << output_path << std::endl;

    return output;
}

cv::Mat renderApprovedPatches(const std::string& target_segment_id, const fs::path& output_path,
                              float opacity, const std::string& patch_filter = "") {
    auto target_meta = vpkg_->loadSurface(target_segment_id);
    QuadSurface* target_surf = target_meta->surface();

    cv::Mat_<cv::Vec3f> raw_points = target_surf->rawPoints();
    cv::Vec2f stored_scale = target_surf->scale();

    // Calculate native resolution
    int native_width = raw_points.cols / stored_scale[0];
    int native_height = raw_points.rows / stored_scale[1];

    std::cout << "Stored points: " << raw_points.cols << "x" << raw_points.rows
              << " at scale " << stored_scale[0] << std::endl;
    std::cout << "Native resolution: " << native_width << "x" << native_height << std::endl;

    float gen_scale = 1.0f;
    cv::Size gen_size(native_width / gen_scale, native_height / gen_scale);

    std::cout << "Generating with scale factor " << gen_scale
              << " (output: " << gen_size.width << "x" << gen_size.height << ")" << std::endl;

    // Generate coordinates
    cv::Mat_<cv::Vec3f> coords;
    cv::Vec3f center = target_surf->pointer();
    cv::Vec3f offset = {
        -(float)(gen_size.width / 2),
        -(float)(gen_size.height / 2),
        0
    };

    target_surf->gen(&coords, nullptr, gen_size, center, gen_scale, offset);

    std::cout << "Generated coords size: " << coords.cols << "x" << coords.rows << std::endl;

    // Load patches
    std::vector<std::string> patch_ids = getOverlapIds(target_segment_id, target_meta, "approved_patches");

    // Parse filter if provided
    std::unordered_set<std::string> filter_set;
    if (!patch_filter.empty()) {
        std::stringstream ss(patch_filter);
        std::string patch_id;
        while (std::getline(ss, patch_id, ',')) {
            // Trim whitespace
            patch_id.erase(0, patch_id.find_first_not_of(" \t"));
            patch_id.erase(patch_id.find_last_not_of(" \t") + 1);
            if (!patch_id.empty()) {
                filter_set.insert(patch_id);
            }
        }
        std::cout << "Filtering to " << filter_set.size() << " specified patches" << std::endl;
    }

    // Load only the patches we want to render
    std::vector<QuadSurface*> patch_surfaces;
    std::vector<std::string> active_patch_ids;  // Track which patches we're actually using

    for (const auto& patch_id : patch_ids) {
        // Skip if filter is active and this patch isn't in the filter
        if (!filter_set.empty() && filter_set.find(patch_id) == filter_set.end()) {
            continue;
        }

        auto patch_meta = vpkg_->loadSurface(patch_id);
        if (patch_meta) {
            patch_surfaces.push_back(patch_meta->surface());
            active_patch_ids.push_back(patch_id);
        }
    }

    std::cout << "Loaded " << patch_surfaces.size() << " patches";
    if (!filter_set.empty()) {
        std::cout << " (filtered from " << patch_ids.size() << " total)";
    }
    std::cout << std::endl;

    // Build spatial index
    std::cout << "Building spatial index..." << std::endl;

    // Estimate good cell size based on patch bounding boxes
    float avg_dimension = 0;
    int count = 0;
    for (auto* patch : patch_surfaces) {
        Rect3D bbox = patch->bbox();
        avg_dimension += (bbox.high[0] - bbox.low[0]);
        avg_dimension += (bbox.high[1] - bbox.low[1]);
        avg_dimension += (bbox.high[2] - bbox.low[2]);
        count += 3;
    }
    float cell_size = (count > 0) ? (avg_dimension / count) * 2.0f : 100.0f;

    MultiSurfaceIndex spatial_index(cell_size);
    for (int i = 0; i < patch_surfaces.size(); i++) {
        spatial_index.addPatch(i, patch_surfaces[i]);
    }

    std::cout << "Spatial index built with " << spatial_index.getCellCount()
              << " cells, cell size: " << cell_size << std::endl;

    // Process with stride
    int stride = 4;
    cv::Size process_size(gen_size.width / stride, gen_size.height / stride);
    cv::Mat_<cv::Vec3b> output_sparse(process_size, cv::Vec3b(255, 255, 255));

    std::cout << "Processing with stride " << stride << ": "
              << process_size.width << "x" << process_size.height << std::endl;

    std::atomic<int> valid_count(0), unmatched_count(0);
    std::vector<std::atomic<int>> patch_counts(patch_surfaces.size());
    for (int i = 0; i < patch_surfaces.size(); i++) {
        patch_counts[i] = 0;
    }

    float tolerance = 1.0f;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < process_size.height; j++) {
        #pragma omp critical
        {
            std::cout << "Processing row " << j << "/" << process_size.height << std::endl;
        }

        for (int i = 0; i < process_size.width; i++) {
            int src_j = j * stride;
            int src_i = i * stride;

            const cv::Vec3f& point = coords(src_j, src_i);

            if (point[0] == -1 || std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2])) {
                continue;
            }

            valid_count++;

            // Get candidate patches from spatial index
            std::vector<int> candidates = spatial_index.getCandidatePatches(point, tolerance);

            bool found_in_patch = false;

            for (int patch_idx : candidates) {
                bool found = patch_surfaces[patch_idx]->containsPoint(point, tolerance);

                if (found) {
                    output_sparse(j, i) = getColormapColor(patch_idx, patch_surfaces.size());
                    patch_counts[patch_idx]++;
                    found_in_patch = true;
                    break;
                }
            }

            if (!found_in_patch) {
                output_sparse(j, i) = cv::Vec3b(0, 0, 0);
                ++unmatched_count;
            }
        }
    }

    cv::imwrite(output_path.string(), output_sparse);

    // Statistics
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Valid points: " << valid_count << std::endl;
    std::cout << "Unmatched: " << unmatched_count
              << " (" << (100.0 * unmatched_count / std::max(1, (int)valid_count)) << "%)" << std::endl;

    int patches_hit = 0;
    for (int i = 0; i < patch_surfaces.size(); i++) {
        if (patch_counts[i] > 0) {
            std::cout << "Patch " << active_patch_ids[i] << ": " << patch_counts[i] << " points" << std::endl;
            patches_hit++;
        }
    }
    std::cout << "Patches with hits: " << patches_hit << "/" << patch_surfaces.size() << std::endl;

    std::cout << "Saved at: " << output_sparse.cols << "x" << output_sparse.rows << std::endl;

    return output_sparse;
}

    AlignmentResult findPatchAlignment(QuadSurface* target_surf, QuadSurface* patch_surf) {
        AlignmentResult result;
        result.valid = false;
        result.num_correspondences = 0;

        cv::Mat_<cv::Vec3f> patch_points = patch_surf->rawPoints();
        std::vector<cv::Vec2f> patch_coords;
        std::vector<cv::Vec2f> target_coords;

        // Sample valid points from patch
        int step = std::max(10, std::min(patch_points.rows, patch_points.cols) / 20);

        for (int j = step; j < patch_points.rows - step; j += step) {
            for (int i = step; i < patch_points.cols - step; i += step) {
                cv::Vec3f point = patch_points(j, i);
                if (point[0] == -1) continue;

                // Find this 3D point in target surface
                cv::Vec3f target_ptr = target_surf->pointer();
                float dist = target_surf->pointTo(target_ptr, point, 2.0, 100);

                if (dist >= 0 && dist <= 2.0) {
                    cv::Vec3f target_loc = target_surf->loc_raw(target_ptr);
                    patch_coords.push_back(cv::Vec2f(i, j));
                    target_coords.push_back(cv::Vec2f(target_loc[0], target_loc[1]));

                    if (patch_coords.size() >= 50) break;
                }
            }
            if (patch_coords.size() >= 50) break;
        }

        if (patch_coords.size() < 5) {
            return result;
        }

        // Compute median offset
        std::vector<float> x_offsets, y_offsets;
        for (size_t k = 0; k < patch_coords.size(); k++) {
            x_offsets.push_back(target_coords[k][0] - patch_coords[k][0]);
            y_offsets.push_back(target_coords[k][1] - patch_coords[k][1]);
        }

        std::sort(x_offsets.begin(), x_offsets.end());
        std::sort(y_offsets.begin(), y_offsets.end());

        result.pixel_offset = cv::Vec2f(
            x_offsets[x_offsets.size() / 2],
            y_offsets[y_offsets.size() / 2]
        );
        result.num_correspondences = patch_coords.size();
        result.valid = true;

        return result;
    }

    cv::Mat renderSequence(const std::string& target_segment_id, const fs::path& output_path, float opacity) {
        std::cout << "Rendering sequence with target segment: " << target_segment_id << std::endl;

        // Load metadata
        auto target_meta = vpkg_->loadSurface(target_segment_id);
        if (!target_meta) {
            throw std::runtime_error("Failed to load target segment: " + target_segment_id);
        }

        fs::path meta_path = target_meta->path / "meta.json";
        std::ifstream meta_file(meta_path);
        json meta_json;
        meta_file >> meta_json;

        std::string seed_id = meta_json["seed"].get<std::string>();
        auto sequence = meta_json["surface_sequence"].get<std::vector<std::string>>();

        std::cout << "Using seed as reference: " << seed_id << std::endl;
        std::cout << "Centering all segments at origin" << std::endl;

        // Load seed
        auto seed_meta = vpkg_->loadSurface(seed_id);
        if (!seed_meta) {
            throw std::runtime_error("Failed to load seed segment: " + seed_id);
        }

        // Find maximum dimensions needed
        QuadSurface* seed_surf = seed_meta->surface();
        int max_width = seed_surf->size().width;
        int max_height = seed_surf->size().height;

        // Check all segments to find max size
        for (const std::string& seq_id : sequence) {
            auto seq_meta = vpkg_->loadSurface(seq_id);
            if (seq_meta) {
                cv::Size s = seq_meta->surface()->size();
                max_width = std::max(max_width, s.width);
                max_height = std::max(max_height, s.height);
            }
            if (seq_id == target_segment_id) break;
        }

        // Create canvas
        cv::Size canvas_size(max_width, max_height);
        cv::Mat_<cv::Vec3b> output(canvas_size, cv::Vec3b(0, 0, 0));
        cv::Mat_<uint8_t> written_mask(canvas_size, (uint8_t)0);

        // Calculate center point of canvas
        int center_x = canvas_size.width / 2;
        int center_y = canvas_size.height / 2;

        // Load seed for reference but don't draw
        auto [seed_image, seed_mask] = loadOrGenerateMaskedImage(seed_surf, seed_meta->path);

        // Calculate total segments for colormap
        int total_segments = sequence.size();
        bool target_in_sequence = false;
        for (const auto& seq_id : sequence) {
            if (seq_id == target_segment_id) {
                target_in_sequence = true;
                break;
            }
        }
        if (!target_in_sequence) {
            total_segments++;
        }

        // Process sequence segments
        for (size_t idx = 0; idx < sequence.size(); idx++) {
            const std::string& seq_id = sequence[idx];
            std::cout << "Processing sequence [" << idx << "]: " << seq_id << std::endl;

            auto seq_meta = vpkg_->loadSurface(seq_id);
            if (!seq_meta) {
                std::cerr << "Failed to load: " << seq_id << std::endl;
                continue;
            }

            QuadSurface* seq_surf = seq_meta->surface();
            auto [seq_image, seq_mask] = loadOrGenerateMaskedImage(seq_surf, seq_meta->path);

            // Center this segment
            int seq_offset_x = center_x - seq_image.cols / 2;
            int seq_offset_y = center_y - seq_image.rows / 2;

            // Get colormap color based on position
            cv::Vec3b color = getColormapColor(idx, total_segments);

            // Draw with write-once
            int pixels_written = 0;
            for (int j = 0; j < seq_image.rows; j++) {
                for (int i = 0; i < seq_image.cols; i++) {
                    if (seq_mask(j, i)) {
                        int out_x = i + seq_offset_x;
                        int out_y = j + seq_offset_y;
                        if (out_x >= 0 && out_x < output.cols && out_y >= 0 && out_y < output.rows) {
                            if (!written_mask(out_y, out_x)) {
                                output(out_y, out_x) = color;
                                written_mask(out_y, out_x) = 1;
                                pixels_written++;
                            }
                        }
                    }
                }
            }

            std::cout << "Added " << pixels_written << " unique pixels" << std::endl;

            if (seq_id == target_segment_id) {
                std::cout << "Reached target segment" << std::endl;
                break;
            }
        }

        // Add target if not in sequence
        if (!target_in_sequence) {
            std::cout << "Adding target segment (not in sequence)" << std::endl;

            QuadSurface* target_surf = target_meta->surface();
            auto [target_image, target_mask] = loadOrGenerateMaskedImage(target_surf, target_meta->path);

            int target_offset_x = center_x - target_image.cols / 2;
            int target_offset_y = center_y - target_image.rows / 2;

            cv::Vec3b color = getColormapColor(sequence.size(), total_segments);

            int pixels_written = 0;
            for (int j = 0; j < target_image.rows; j++) {
                for (int i = 0; i < target_image.cols; i++) {
                    if (target_mask(j, i)) {
                        int out_x = i + target_offset_x;
                        int out_y = j + target_offset_y;
                        if (out_x >= 0 && out_x < output.cols && out_y >= 0 && out_y < output.rows) {
                            if (!written_mask(out_y, out_x)) {
                                output(out_y, out_x) = color;
                                written_mask(out_y, out_x) = 1;
                                pixels_written++;
                            }
                        }
                    }
                }
            }
            std::cout << "Target added " << pixels_written << " unique pixels" << std::endl;
        }

        // Auto-crop
        cv::Mat cropped = autoCrop(output, written_mask);

        cv::imwrite(output_path.string(), cropped);
        std::cout << "Saved to: " << output_path << std::endl;

        return cropped;
    }

    std::vector<std::string> getOverlapIds(const std::string& root_id,
                                          std::shared_ptr<SurfaceMeta> root_meta,
                                          std::string source) {
        std::vector<std::string> overlap_ids;

        if (source == "overlapping") {
            root_meta->readOverlapping();
            if (root_meta->overlapping_str.empty()) {
                throw std::runtime_error("No overlapping segments found in overlapping.json for segment: " + root_id);
            }
            overlap_ids.assign(root_meta->overlapping_str.begin(), root_meta->overlapping_str.end());
        } else if (source == "contributing")  {
            fs::path meta_path = root_meta->path / "meta.json";
            if (!fs::exists(meta_path)) {
                throw std::runtime_error("meta.json not found for segment: " + root_id);
            }

            std::ifstream meta_file(meta_path);
            json meta_json;
            meta_file >> meta_json;

            if (!meta_json.contains("contributing_surfaces")) {
                throw std::runtime_error("contributing_surfaces not found in meta.json for segment: " + root_id);
            }

            overlap_ids = meta_json["contributing_surfaces"].get<std::vector<std::string>>();
            if (overlap_ids.empty()) {
                throw std::runtime_error("contributing_surfaces is empty in meta.json for segment: " + root_id);
            }
        } else if (source == "approved_patches") {
            fs::path meta_path = root_meta->path / "meta.json";
            if (!fs::exists(meta_path)) {
                throw std::runtime_error("meta.json not found for segment: " + root_id);
            }

            std::ifstream meta_file(meta_path);
            json meta_json;
            meta_file >> meta_json;

            if (!meta_json.contains("used_approved_segments")) {
                throw std::runtime_error("used_approved_segments not found in meta.json for segment: " + root_id);
            }

            overlap_ids = meta_json["used_approved_segments"].get<std::vector<std::string>>();
            if (overlap_ids.empty()) {
                throw std::runtime_error("used_approved_segments is empty in meta.json for segment: " + root_id);
            }
        }

        return overlap_ids;
    }

    std::pair<cv::Mat_<cv::Vec3b>, cv::Mat_<uint8_t>> loadOrGenerateMaskedImage(
        QuadSurface* surf, const fs::path& segment_path) {

        cv::Mat_<uint8_t> mask;
        cv::Mat_<uint8_t> img;
        fs::path mask_path = segment_path / "mask.tif";

        if (mask.empty()) {
            std::cout << "Generating mask and image data" << std::endl;

            z5::Dataset* ds_high = volume_->zarrDataset(0);
            z5::Dataset* ds_low = nullptr;
            if (volume_->numScales() > 2) {
                ds_low = volume_->zarrDataset(2);
            }

            generate_mask(surf, mask, img, ds_high, ds_low, cache_);

            if (!mask_path.parent_path().empty()) {
                cv::imwrite(mask_path.string(), mask);
                std::cout << "Saved generated mask to: " << mask_path << std::endl;
            }
        } else {
            std::cout << "Generating image data for existing mask" << std::endl;

            cv::Size native_size = surf->size();
            cv::Mat_<cv::Vec3f> coords;
            cv::Vec3f center = surf->pointer();
            surf->gen(&coords, nullptr, native_size, center, 1.0f, {0, 0, 0});

            int ds_idx = 0;
            if (native_size.width >= 4000 && volume_->numScales() > 2) {
                ds_idx = 2;
            } else if (native_size.width >= 2000 && volume_->numScales() > 1) {
                ds_idx = 1;
            }

            float ds_scale = std::pow(2.0f, -ds_idx);
            readInterpolated3D(img, volume_->zarrDataset(ds_idx), coords * ds_scale, cache_);

            if (ds_idx > 0) {
                cv::resize(img, img, native_size, 0, 0, cv::INTER_LINEAR);
            }
        }

        cv::Mat_<cv::Vec3b> rgb_image;
        cv::cvtColor(img, rgb_image, cv::COLOR_GRAY2RGB);

        for (int y = 0; y < rgb_image.rows; y++) {
            for (int x = 0; x < rgb_image.cols; x++) {
                if (!mask(y, x)) {
                    rgb_image(y, x) = cv::Vec3b(0, 0, 0);
                }
            }
        }

        return {rgb_image, mask};
    }

    AlignmentResult findAlignment(QuadSurface* ref_surf, QuadSurface* target_surf) {
        AlignmentResult result;
        result.valid = false;
        result.num_correspondences = 0;

        QuadSurface* intersection = surface_intersection(ref_surf, target_surf, 2.0);
        if (!intersection) {
            return result;
        }

        cv::Mat_<cv::Vec3f> intersect_points = intersection->rawPoints();
        std::vector<cv::Vec2f> ref_coords;
        std::vector<cv::Vec2f> target_coords;

        int step = std::max(5, std::min(intersect_points.rows, intersect_points.cols) / 30);

        for (int j = step; j < intersect_points.rows - step; j += step) {
            for (int i = step; i < intersect_points.cols - step; i += step) {
                cv::Vec3f point = intersect_points(j, i);
                if (point[0] == -1) continue;

                cv::Vec3f ref_ptr = ref_surf->pointer();
                cv::Vec3f target_ptr = target_surf->pointer();

                float ref_dist = ref_surf->pointTo(ref_ptr, point, 2.0, 1000);
                float target_dist = target_surf->pointTo(target_ptr, point, 2.0, 1000);

                if (ref_dist >= 0 && ref_dist <= 2.0 && target_dist >= 0 && target_dist <= 2.0) {
                    cv::Vec3f ref_loc = ref_surf->loc_raw(ref_ptr);
                    cv::Vec3f target_loc = target_surf->loc_raw(target_ptr);

                    ref_coords.push_back(cv::Vec2f(ref_loc[0], ref_loc[1]));
                    target_coords.push_back(cv::Vec2f(target_loc[0], target_loc[1]));
                }

                if (ref_coords.size() >= 100) break;
            }
            if (ref_coords.size() >= 100) break;
        }

        delete intersection;

        if (ref_coords.size() < 10) {
            return result;
        }

        std::vector<float> x_offsets, y_offsets;
        for (size_t k = 0; k < ref_coords.size(); k++) {
            x_offsets.push_back(ref_coords[k][0] - target_coords[k][0]);
            y_offsets.push_back(ref_coords[k][1] - target_coords[k][1]);
        }

        std::sort(x_offsets.begin(), x_offsets.end());
        std::sort(y_offsets.begin(), y_offsets.end());

        int trim_count = x_offsets.size() / 5;
        float x_sum = 0, y_sum = 0;
        int count = 0;
        for (size_t i = trim_count; i < x_offsets.size() - trim_count; i++) {
            x_sum += x_offsets[i];
            y_sum += y_offsets[i];
            count++;
        }

        result.pixel_offset = cv::Vec2f(x_sum / count, y_sum / count);
        result.num_correspondences = ref_coords.size();
        result.valid = true;

        return result;
    }

    cv::Mat autoCrop(const cv::Mat_<cv::Vec3b>& image, const cv::Mat_<uint8_t>& mask) {
        int min_x = image.cols, max_x = 0;
        int min_y = image.rows, max_y = 0;

        for (int y = 0; y < mask.rows; y++) {
            for (int x = 0; x < mask.cols; x++) {
                if (mask(y, x)) {
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                    min_y = std::min(min_y, y);
                    max_y = std::max(max_y, y);
                }
            }
        }

        if (min_x > max_x || min_y > max_y) {
            return image;
        }

        cv::Rect crop_rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
        return image(crop_rect).clone();
    }
};

int main(int argc, char* argv[]) {
    if (argc < 6 || argc > 7) {
        std::cout << "Usage: " << argv[0] << " <volpkg-path> <volume-id> <segment-id> <overlap-source> <output-png> [patch-filter]" << std::endl;
        std::cout << "  volpkg-path: Path to volume package" << std::endl;
        std::cout << "  volume-id: ID of volume to use" << std::endl;
        std::cout << "  segment-id: ID of segment to render" << std::endl;
        std::cout << "  overlap-source: Source for overlaps (overlapping|contributing|sequence|approved_patches|duplicate_check)" << std::endl;
        std::cout << "  output-png: Output file path" << std::endl;
        std::cout << "  patch-filter: (optional) Comma-separated list of patch IDs to render (only for approved_patches)" << std::endl;
        return EXIT_SUCCESS;
    }

    fs::path volpkg_path = argv[1];
    std::string volume_id = argv[2];
    std::string segment_id = argv[3];
    std::string overlap_source = argv[4];
    fs::path output_path = argv[5];
    float opacity = 0.5f;

    std::string patch_filter = "";
    if (argc == 7) {
        patch_filter = argv[6];
    }

    if (overlap_source != "overlapping" && overlap_source != "sequence" &&
    overlap_source != "contributing" && overlap_source != "approved_patches" &&
    overlap_source != "duplicate_check") {
        std::cerr << "Error: Invalid overlap source. Must be one of: overlapping, contributing, sequence, approved_patches, duplicate_check" << std::endl;
        return EXIT_FAILURE;
    }

    if (opacity < 0.0f || opacity > 1.0f) {
        std::cerr << "Error: Opacity must be between 0.0 and 1.0" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        SegmentRenderer renderer(volpkg_path, volume_id);
        renderer.render(segment_id, output_path, overlap_source, opacity, patch_filter);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}