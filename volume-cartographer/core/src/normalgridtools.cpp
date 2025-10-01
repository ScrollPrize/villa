#include "vc/core/util/GridStore.hpp"
#include <random>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "vc/core/util/normalgridtools.hpp"

namespace vc::core::util {

long total_candidates_before_dedup = 0;
long total_candidates_after_dedup = 0;
long nearest_neighbors_calls = 0;

cv::Vec2f align_and_extract_umbilicus(const GridStore& grid_store) {
    auto segments = grid_store.get_all();
    if (segments.empty()) {
        return cv::Vec2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    }

    // 1. Define RANSAC parameters
    const int num_ransac_iterations = 1000;
    const int num_samples_per_iteration = 10000;

    // 2. Determine grid extents for sampling umbilicus candidates
    cv::Size grid_size = grid_store.size();
    float min_x = 0;
    float max_x = grid_size.width;
    float min_y = 0;
    float max_y = grid_size.height;

    // 3. Create a global list of all line segments
    std::vector<std::pair<cv::Point, cv::Point>> all_line_segments;
    for (const auto& path : segments) {
        if (path->size() < 2) continue;
        for (size_t i = 0; i < path->size() - 1; ++i) {
            all_line_segments.push_back(std::make_pair((*path)[i], (*path)[i+1]));
        }
    }

    if (all_line_segments.empty()) {
        return cv::Vec2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    }

    // 4. Pre-sample points and calculate normals for consistent scoring
    std::vector<cv::Point2f> sample_points;
    std::vector<cv::Vec2f> sample_normals;
    sample_points.reserve(num_samples_per_iteration);
    sample_normals.reserve(num_samples_per_iteration);

    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> segment_dist(0, all_line_segments.size() - 1);

    for (int j = 0; j < num_samples_per_iteration; ++j) {
        const auto& line_segment = all_line_segments[segment_dist(gen)];
        const auto& p1 = line_segment.first;
        const auto& p2 = line_segment.second;

        sample_points.emplace_back((float)(p1.x + p2.x) / 2.0f, (float)(p1.y + p2.y) / 2.0f);
        
        cv::Vec2f tangent((float)(p2.x - p1.x), (float)(p2.y - p1.y));
        cv::normalize(tangent, tangent);
        sample_normals.emplace_back(-tangent[1], tangent[0]); // Perpendicular
    }

    // 4. RANSAC loop
    cv::Vec2f best_umbilicus(0, 0);
    double best_score = -1.0;

    std::uniform_real_distribution<float> x_dist(min_x, max_x);
    std::uniform_real_distribution<float> y_dist(min_y, max_y);

    for (int i = 0; i < num_ransac_iterations; ++i) {
        // a. Sample a candidate umbilicus point
        cv::Vec2f candidate_umbilicus(x_dist(gen), y_dist(gen));

        // b. Score the candidate
        double current_score = 0.0;
        for (size_t j = 0; j < sample_points.size(); ++j) {
            const auto& point = sample_points[j];
            const auto& normal = sample_normals[j];

            cv::Vec2f umbilicus_to_segment = cv::Vec2f(point) - candidate_umbilicus;
            cv::normalize(umbilicus_to_segment, umbilicus_to_segment);

            double cos_angle = umbilicus_to_segment.dot(normal);
            current_score += cos_angle * cos_angle;
        }

        // c. Update best estimate
        if (current_score > best_score) {
            best_score = current_score;
            best_umbilicus = candidate_umbilicus;
        }
    }

    // 5. Refine the best estimate using a hill-climbing direct search.
    auto score_candidate = [&](const cv::Vec2f& candidate) {
        double score = 0.0;
        double wsum = 0.0;
        for (size_t j = 0; j < sample_points.size(); ++j) {
            const auto& point = sample_points[j];
            const auto& normal = sample_normals[j];

            cv::Vec2f umbilicus_to_segment = cv::Vec2f(point) - candidate;
            float dist = cv::norm(umbilicus_to_segment);
            if (dist < 1e-6) continue;

            umbilicus_to_segment /= dist; // Manual normalization

            double cos_angle = umbilicus_to_segment.dot(normal);
            float weight = 1.0f / std::max(100.0f, dist);
            score += (cos_angle * cos_angle) * weight;
            wsum += weight;
        }
        return score/wsum;
    };

    double refined_best_score = score_candidate(best_umbilicus);
    float step_size = 1024.0f;

    while (step_size >= 1.0f) {
        bool moved = false;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) continue;

                cv::Vec2f candidate = best_umbilicus + cv::Vec2f(dx * step_size, dy * step_size);
                double candidate_score = score_candidate(candidate);

                if (candidate_score > refined_best_score) {
                    refined_best_score = candidate_score;
                    best_umbilicus = candidate;
                    moved = true;
                }
            }
        }

        if (!moved) {
            step_size /= 2.0f;
        }
    }

    std::cout << "Refined umbilicus estimate: " << best_umbilicus << " with score " << refined_best_score << std::endl;

    return best_umbilicus;
}

void align_and_filter_segments(const GridStore& grid_store, GridStore& result, const cv::Vec2f& center_point) {
    cv::Rect rect(0, 0, grid_store.size().width, grid_store.size().height);
    int grid_step = 64;

    SegmentGrid all(rect, grid_step);
    SegmentGrid unused(rect, grid_step);
    SegmentGrid assigned(rect, grid_step);

    auto segments = grid_store.get_all();
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& segment = *segments[i];
        if (segment.size() < 2) continue;
        for (size_t j = 0; j < segment.size() - 1; ++j) {
            auto new_segment = std::make_shared<SegmentInfo>(segment[j], segment[j+1], i, j);
            all.add(new_segment);
            unused.add(new_segment);
        }
    }

    // Find the best initial segment
    const int num_initial_samples = 100;
    std::shared_ptr<SegmentInfo> best_initial_segment = nullptr;
    double best_initial_score = -1.0;

    for (int i = 0; i < num_initial_samples; ++i) {
        auto candidate = unused.get_random_segment();
        if (!candidate) continue;

        auto neighbors = unused.nearest_neighbors(candidate->middle_point, 20);
        double score = 0.0;
        for (const auto& neighbor : neighbors) {
            if (neighbor == candidate) continue;
            double cos_angle = std::abs(candidate->normal.dot(neighbor->normal));
            score += cos_angle;
        }

        if (score > best_initial_score) {
            best_initial_score = score;
            best_initial_segment = candidate;
        }
    }

    if (best_initial_segment) {
        assigned.add(best_initial_segment);
        unused.remove(best_initial_segment);
    }

    // Main iterative alignment loop
    int iter = 0;
    while (true) {
        iter++;
        // if (iter == 10000)
            // break;
        if (unused.count() == 0) {
            std::cout << "Iteration " << iter << ": No unused segments left. Exiting." << std::endl;
            break;
        }
        // std::cout << "\n--- Iteration " << iter << " ---" << std::endl;
        // std::cout << "Assigned: " << assigned.count() << ", Unused: " << unused.count() << std::endl;

        // 1. Sample a set of random points from assigned
        auto assigned_sample = assigned.get_random_segment();
        if (!assigned_sample) continue;

        // 2. For each of those get 20 knn points from unused
        auto candidates = unused.nearest_neighbors(assigned_sample->middle_point, 20);
        // std::cout << "Found " << candidates.size() << " candidates near assigned sample." << std::endl;

        // 3. Find the candidate with the densest neighborhood of assigned segments
        std::shared_ptr<SegmentInfo> best_candidate = nullptr;
        float min_max_dist = std::numeric_limits<float>::max();

        for (const auto& candidate : candidates) {
            // std::cout << "retrieve from assigned points " << std::endl;
            auto assigned_neighbors = assigned.nearest_neighbors(candidate->middle_point, 10);
            // std::cout << "done " << std::endl;

            if (!assigned_neighbors.size())
                continue;

            float max_dist = 0.0f;
            for(const auto& neighbor : assigned_neighbors) {
                max_dist = std::max(max_dist, (float)cv::norm(candidate->middle_point - neighbor->middle_point));
            }

            max_dist *= 10.0/assigned_neighbors.size()+100.0;

            if (max_dist < min_max_dist) {
                min_max_dist = max_dist;
                best_candidate = candidate;
            }
        }

        if (!best_candidate) {
            // std::cout << "No best candidate found in this iteration." << std::endl;
            continue;
        }
        // std::cout << "Best candidate found with min_max_dist: " << min_max_dist << std::endl;

        // 4. Calculate scores for flipping vs. not flipping
        auto neighbors = assigned.nearest_neighbors(best_candidate->middle_point, 20);
        double score_no_flip = 0.0;
        double score_flip = 0.0;

        for (const auto& neighbor : neighbors) {
            float dist = cv::norm(best_candidate->middle_point - neighbor->middle_point);
            float weight = 1.0f / std::max(1.0f, dist);

            score_no_flip += best_candidate->normal.dot(neighbor->normal) * weight;
            score_flip += (-best_candidate->normal).dot(neighbor->normal) * weight;
        }

        // 5. Add the best candidate with the correct orientation to assigned
        // std::cout << "Scores - No Flip: " << score_no_flip << ", Flip: " << score_flip << std::endl;
        if (score_flip > score_no_flip) {
            // std::cout << "Flipping candidate." << std::endl;
            best_candidate->flipped = true;
            best_candidate->normal *= -1;
        }
        assigned.add(best_candidate);
        unused.remove(best_candidate);


        // if (iter % 1000 == 0) {
        //     if (nearest_neighbors_calls > 0) {
        //         double avg_before = static_cast<double>(total_candidates_before_dedup) / nearest_neighbors_calls;
        //         double avg_after = static_cast<double>(total_candidates_after_dedup) / nearest_neighbors_calls;
        //         std::cout << "Iteration " << iter << ", Avg. candidates before dedup: " << avg_before
        //                   << ", after dedup: " << avg_after << std::endl;
        //     } else {
        //         std::cout << "Iteration " << iter << std::endl;
        //     }
        //     // 6. Debug visualization
        //     // GridStore tmp(rect, grid_step);
        //     // convert_segment_grid_to_grid_store(assigned, grid_store, tmp);
        //     //
        //     // cv::Mat vis = visualize_segment_directions(tmp);
        //     //
        //     // char filename[256];
        //     // snprintf(filename, sizeof(filename), "assigned_iter_%03d.tif", iter);
        //     // cv::imwrite(filename, vis);
        // }
    }


    // Rebuild the GridStore from the aligned segments
    // Optional: Final global alignment check based on the umbilicus
    if (!std::isnan(center_point[0]) && !std::isnan(center_point[1])) {
        double alignment_score = 0.0;
        const int num_samples = 1000;
        for (int i = 0; i < num_samples; ++i) {
            auto seg = assigned.get_random_segment();
            if (!seg) continue;

            cv::Vec2f center_to_segment = seg->middle_point - cv::Point2f(center_point);
            cv::normalize(center_to_segment, center_to_segment);
            
            double dot_product = seg->normal.dot(center_to_segment);
            alignment_score += dot_product * std::abs(dot_product);
        }

        if (alignment_score < 0) {
            // The normals are predominantly pointing inwards, so flip them all
            for (auto& seg : assigned.get_all_segments()) {
                seg->flipped = !seg->flipped;
            }
        }
        result.meta["umbilicus_x"] = center_point[0];
        result.meta["umbilicus_y"] = center_point[1];
    }

    // Rebuild the GridStore from the aligned segments
    convert_segment_grid_to_grid_store(assigned, grid_store, result);
    result.meta["aligned"] = true;

    // Final visualization of the result
    // cv::Mat final_vis = visualize_segment_directions(result);
    // cv::imwrite("final_aligned_segments.tif", final_vis);
}

cv::Mat visualize_segment_directions(const GridStore& grid_store) {
    cv::Size grid_size = grid_store.size();
    cv::Mat vis = cv::Mat::zeros(grid_size.height > 0 ? grid_size.height : 1024, grid_size.width > 0 ? grid_size.width : 1024, CV_8UC3);

    auto segments = grid_store.get_all();
    for (const auto& segment_ptr : segments) {
        const auto& segment = *segment_ptr;
        if (segment.size() < 2) continue;

        for (size_t i = 0; i < segment.size() - 1; ++i) {
            const auto& p1 = segment[i];
            const auto& p2 = segment[i+1];

            cv::Point center = (p1 + p2) / 2;
            cv::Vec2f tangent((float)(p2.x - p1.x), (float)(p2.y - p1.y));
            cv::normalize(tangent, tangent);
            cv::Vec2f normal(-tangent[1], tangent[0]);

            cv::Point endpoint(center.x + normal[0] * 20, center.y + normal[1] * 20);
            cv::Point normal_endpoint(center.x + normal[0] * 5, center.y + normal[1] * 5);

            cv::circle(vis, center, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA); // Red dot at the base
            cv::arrowedLine(vis, center, endpoint, cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0, 0.3); // White arrow for direction
            cv::line(vis, center, normal_endpoint, cv::Scalar(0, 255, 0), 1, cv::LINE_AA); // Green line for normal
        }
    }

    return vis;
}

void convert_segment_grid_to_grid_store(const SegmentGrid& segment_grid, const GridStore& original_grid_store, GridStore& grid_store) {
    auto original_paths = original_grid_store.get_all();

    // 1. Get all segments and store them in a map by original path index
    std::unordered_map<size_t, std::vector<std::shared_ptr<SegmentInfo>>> path_segments;
    for (const auto& seg : segment_grid.get_all_segments()) {
        path_segments[seg->original_path_idx].push_back(seg);
    }

    // 2. For each original path, sort its segments and build new sub-paths
    for (auto& pair : path_segments) {
        auto& segments_for_path = pair.second;
        std::sort(segments_for_path.begin(), segments_for_path.end(),
            [](const auto& a, const auto& b) {
            return a->original_segment_idx < b->original_segment_idx;
        });

        if (segments_for_path.empty()) continue;

        std::vector<cv::Point> current_sub_path;
        bool current_flip_status = segments_for_path[0]->flipped;

        bool has_discontinuity = false;
        for (size_t i = 1; i < segments_for_path.size(); ++i) {
            if (segments_for_path[i]->original_segment_idx != segments_for_path[i-1]->original_segment_idx + 1) {
                has_discontinuity = true;
                break;
            }
        }

        if (has_discontinuity) {
            std::cerr << "Skipping discontinuous path with original_path_idx " << pair.first << std::endl;
            continue;
        }

        for (const auto& seg : segments_for_path) {
            if (seg->flipped != current_flip_status) {
                if (current_sub_path.size() >= 2) {
                    if (current_flip_status) {
                        std::reverse(current_sub_path.begin(), current_sub_path.end());
                    }
                    grid_store.add(current_sub_path);
                }
                current_sub_path.clear();
                current_flip_status = seg->flipped;
            }

            const auto& original_path = *original_paths[seg->original_path_idx];
            const auto& p1 = original_path[seg->original_segment_idx];
            const auto& p2 = original_path[seg->original_segment_idx + 1];

            if (current_sub_path.empty()) {
                current_sub_path.push_back(p1);
            }
            current_sub_path.push_back(p2);
        }

        // Add the last sub-path
        if (current_sub_path.size() >= 2) {
            if (current_flip_status) {
                std::reverse(current_sub_path.begin(), current_sub_path.end());
            }
            grid_store.add(current_sub_path);
        }
    }
}


SegmentGrid::SegmentGrid(const cv::Rect& rect, int grid_step)
    : rect(rect), grid_step(grid_step), gen(std::random_device()()) {
    int grid_width = (rect.width + grid_step - 1) / grid_step;
    int grid_height = (rect.height + grid_step - 1) / grid_step;
    grid.resize(grid_height, std::vector<std::vector<std::weak_ptr<SegmentInfo>>>(grid_width));
}

void SegmentGrid::add(const std::shared_ptr<SegmentInfo>& segment) {
    all_segments.push_back(segment);

    int grid_x = (segment->middle_point.x - rect.x) / grid_step;
    int grid_y = (segment->middle_point.y - rect.y) / grid_step;
    if (grid_y >= 0 && grid_y < grid.size() && grid_x >= 0 && grid_x < grid[0].size()) {
        grid[grid_y][grid_x].push_back(segment);
    }
}

void SegmentGrid::remove(const std::shared_ptr<SegmentInfo>& segment_to_remove) {
    // Find the segment in the main vector
    auto it = std::find(all_segments.begin(), all_segments.end(), segment_to_remove);
    if (it == all_segments.end()) return;

    // Swap with the last element and pop back for O(1) removal
    std::swap(*it, all_segments.back());
    all_segments.pop_back();

    // Remove from the grid
    int grid_x = (segment_to_remove->middle_point.x - rect.x) / grid_step;
    int grid_y = (segment_to_remove->middle_point.y - rect.y) / grid_step;
    if (grid_y >= 0 && grid_y < grid.size() && grid_x >= 0 && grid_x < grid[0].size()) {
        auto& cell = grid[grid_y][grid_x];
        cell.erase(std::remove_if(cell.begin(), cell.end(),
            [&](const std::weak_ptr<SegmentInfo>& weak_ptr) {
                return weak_ptr.expired() || weak_ptr.lock() == segment_to_remove;
            }), cell.end());
    }
}

std::vector<std::shared_ptr<SegmentInfo>> SegmentGrid::nearest_neighbors(const cv::Point2f& point, int n) const {
    if (all_segments.empty()) {
        return {};
    }

    int center_grid_x = (point.x - rect.x) / grid_step;
    int center_grid_y = (point.y - rect.y) / grid_step;

    std::vector<std::shared_ptr<SegmentInfo>> candidates;
    candidates.reserve(n * 2);

    if (n >= count())
        return get_all_segments();

    int radius = -1;
    bool enough = false;
    while (!enough) {
        radius++;

        if (candidates.size() > n)
            enough = true;
        

        for (int y = center_grid_y - radius-1; y <= center_grid_y + radius+1; ++y) {
            for (int x = center_grid_x - radius-1; x <= center_grid_x + radius+1; ++x) {
                // Euclidean distance check in grid units
                float point_grid_x = (point.x - rect.x) / grid_step;
                float point_grid_y = (point.y - rect.y) / grid_step;
                float dx = (x + 0.5f) - point_grid_x;
                float dy = (y + 0.5f) - point_grid_y;
                float dist_sq = dx * dx + dy * dy;
                float radius_sq = (radius + 1) * (radius + 1);

                if (dist_sq > radius_sq) {
                    continue;
                }
                if (radius > 0) {
                    float prev_radius_sq = radius * radius;
                    if (dist_sq <= prev_radius_sq) {
                        continue;
                    }
                }

                if (y >= 0 && y < grid.size() && x >= 0 && x < grid[0].size()) {
                    for (const auto& weak_ptr : grid[y][x]) {
                        if (auto shared_ptr = weak_ptr.lock()) {
                            candidates.push_back(shared_ptr);
                            // added_in_this_ring = true;
                        }
                    }
                }
            }
        }
    }

    // Pre-calculate squared distances for efficient sorting
    std::vector<std::pair<float, std::shared_ptr<SegmentInfo>>> sorted_candidates;
    sorted_candidates.reserve(candidates.size());
    for (const auto& seg : candidates) {
        cv::Point2f diff = seg->middle_point - point;
        sorted_candidates.push_back({diff.dot(diff), seg});
    }

    // Sort candidates by pre-calculated squared distance
    std::sort(sorted_candidates.begin(), sorted_candidates.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

    // Re-populate candidates vector with sorted segments
    candidates.clear();
    for (const auto& pair : sorted_candidates) {
        candidates.push_back(pair.second);
    }

    total_candidates_before_dedup += candidates.size();

    total_candidates_after_dedup += candidates.size();
    nearest_neighbors_calls++;

    // Return top N
    if (candidates.size() > n) {
        candidates.resize(n);
    }

    return candidates;
}

std::shared_ptr<SegmentInfo> SegmentGrid::get_random_segment() {
    if (all_segments.empty()) {
        return nullptr;
    }
    std::uniform_int_distribution<> dist(0, all_segments.size() - 1);
    return all_segments[dist(gen)];
}

size_t SegmentGrid::count() const {
    return all_segments.size();
}

}
