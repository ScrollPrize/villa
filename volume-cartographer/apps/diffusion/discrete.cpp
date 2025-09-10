#include "discrete.hpp"
#include "support.hpp"
#include "spiral_common.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <chrono>
#include <unordered_set>
#include <iomanip>
#include <random>

#include <vc/ui/VCCollection.hpp>

constexpr int32_t BASE_LABEL_STEP = 4096;

struct ConflictGroup {
    int size;
    int32_t label1;
    int32_t label2;
    int offset;
};

void visualize_labels(
    const cv::Mat& labels,
    const std::unordered_map<int32_t, std::string>& label_to_name_map,
    const std::unordered_map<std::string, int>& collection_offsets,
    const fs::path& output_path
);

void visualize_conflicts(
    const cv::Mat& labels,
    const std::vector<std::tuple<cv::Point, int32_t, int32_t>>& conflicts,
    const std::unordered_map<int32_t, std::string>& label_to_name_map,
    const std::unordered_map<std::string, int>& collection_offsets,
    const fs::path& conflicts_path
);

std::vector<ConflictGroup> group_and_sort_conflicts(
    const std::vector<std::tuple<cv::Point, int32_t, int32_t>>& conflicts
);

std::string get_collection_name_from_label(int32_t label, const std::unordered_map<int32_t, std::string>& map) {
    if (label == 0) return "unlabeled";
    int32_t base_label = (label / BASE_LABEL_STEP) * BASE_LABEL_STEP;
    if (map.count(base_label)) {
        return map.at(base_label);
    }
    throw std::runtime_error("Could not find collection name for label " + std::to_string(label));
}

void print_conflict_report(
    const std::vector<ConflictGroup>& all_groups,
    const std::unordered_map<int32_t, std::string>& label_to_name_map,
    const std::unordered_map<std::string, int>& collection_offsets
) {
    std::cout << "\n--- Conflict Report ---" << std::endl;
    int false_conflicts = 0;
    int self_conflicts = 0;
    for (const auto& group : all_groups) {
        const std::string name1 = get_collection_name_from_label(group.label1, label_to_name_map);
        const std::string name2 = get_collection_name_from_label(group.label2, label_to_name_map);

        if (name1 == name2) {
            self_conflicts++;
        }

        if (collection_offsets.count(name1) && collection_offsets.count(name2)) {
            int corrected_offset = collection_offsets.at(name1) - collection_offsets.at(name2);
            if (group.offset == corrected_offset) {
                false_conflicts++;
            }
        }
    }

    std::cout << "Total conflict groups: " << all_groups.size() << std::endl;
    std::cout << "Self-conflict groups: " << self_conflicts << std::endl;
    if (!collection_offsets.empty()) {
        std::cout << "False conflict groups (explained by offsets): " << false_conflicts << std::endl;
    }

    std::cout << "\n--- Self-Conflict Groups ---" << std::endl;
    std::cout << "Size, Collection, Implied Offset" << std::endl;
    for (const auto& group : all_groups) {
        const std::string name1 = get_collection_name_from_label(group.label1, label_to_name_map);
        const std::string name2 = get_collection_name_from_label(group.label2, label_to_name_map);
        if (name1 == name2) {
            std::cout << group.size << ", " << name1 << ", " << group.offset << std::endl;
        }
    }

    std::cout << "\n--- Unexplained Cross-Conflict Groups ---" << std::endl;
    std::cout << "Size, Collection A, Collection B, Implied Offset, Corrected Offset" << std::endl;
    for (const auto& group : all_groups) {
        const std::string name1 = get_collection_name_from_label(group.label1, label_to_name_map);
        const std::string name2 = get_collection_name_from_label(group.label2, label_to_name_map);
        if (name1 == name2) continue;

        if (collection_offsets.count(name1) && collection_offsets.count(name2)) {
            int corrected_offset = collection_offsets.at(name1) - collection_offsets.at(name2);
            if (group.offset != corrected_offset) {
                std::cout << group.size << ", "
                << name1 << ", " << name2 << ", "
                << group.offset << ", " << corrected_offset << std::endl;
            }
        } else {
            std::cout << group.size << ", "
            << name1 << ", " << name2 << ", "
            << group.offset << ", " << "N/A" << std::endl;
        }
    }
}

std::unordered_map<std::string, int> estimate_offsets(
    std::vector<ConflictGroup>& all_groups,
    const std::unordered_map<int32_t, std::string>& label_to_name_map
) {
    std::unordered_map<std::string, int> collection_offsets;
    if (label_to_name_map.empty() || all_groups.empty()) {
        std::cout << "\n--- No conflicts or labels found for offset estimation. ---" << std::endl;
        return collection_offsets;
    }

    // Initialize collection sets - each collection is its own set initially.
    std::unordered_map<std::string, std::string> collection_to_set;
    std::unordered_map<std::string, std::vector<std::string>> set_to_collections;
    for (const auto& [label, name] : label_to_name_map) {
        collection_offsets[name] = 0;
        collection_to_set[name] = name;
        set_to_collections[name] = {name};
    }

    while (true) {
        // Aggregate evidence between sets
        std::map<std::pair<std::string, std::string>, std::map<int, int>> evidence_map;
        for (const auto& group : all_groups) {
            std::string name1 = get_collection_name_from_label(group.label1, label_to_name_map);
            std::string name2 = get_collection_name_from_label(group.label2, label_to_name_map);

            std::string set1 = collection_to_set.at(name1);
            std::string set2 = collection_to_set.at(name2);

            if (set1 == set2) continue;

            int offset1 = collection_offsets.at(name1);
            int offset2 = collection_offsets.at(name2);
            int implied_offset = group.offset;

            // The implied offset between sets is offset1 - offset2 = implied_offset
            // So, the offset to add to set2 to align with set1 is:
            // offset_to_set2 = offset1 - offset2 - implied_offset
            int set_offset = (offset1 - offset2) - implied_offset;

            if (set1 > set2) {
                std::swap(set1, set2);
                set_offset = -set_offset;
            }
            evidence_map[{set1, set2}][set_offset] += group.size;
        }

        if (evidence_map.empty()) {
            std::cout << "No more evidence between sets. Stopping." << std::endl;
            break;
        }

        // Find the best evidence
        int max_pixels = 0;
        std::string best_set1, best_set2;
        int best_offset = 0;

        for (const auto& [set_pair, offset_map] : evidence_map) {
            for (const auto& [offset, pixels] : offset_map) {
                if (pixels > max_pixels) {
                    max_pixels = pixels;
                    best_set1 = set_pair.first;
                    best_set2 = set_pair.second;
                    best_offset = offset;
                }
            }
        }

        if (max_pixels == 0) {
            std::cout << "No single best evidence found. Stopping." << std::endl;
            break;
        }

        // Merge the sets
        std::cout << "Merging set '" << best_set2 << "' into '" << best_set1
        << "' with offset " << best_offset << " (evidence: " << max_pixels << " pixels)" << std::endl;

        // Apply offset to all collections in the second set
        for (const auto& name_to_update : set_to_collections.at(best_set2)) {
            collection_offsets.at(name_to_update) += best_offset;
            collection_to_set.at(name_to_update) = best_set1;
        }

        // Move collections from the merged set to the target set
        set_to_collections.at(best_set1).insert(
            set_to_collections.at(best_set1).end(),
                                                set_to_collections.at(best_set2).begin(),
                                                set_to_collections.at(best_set2).end()
        );
        set_to_collections.erase(best_set2);

        if (set_to_collections.size() == 1) {
            std::cout << "All collections merged into a single set." << std::endl;
            break;
        }
    }

    std::cout << "\n--- Estimated Collection Offsets ---" << std::endl;
    for (auto const& [name, offset] : collection_offsets) {
        std::cout << name << ": " << offset << std::endl;
    }
    for (const auto& [label, name] : label_to_name_map) {
        if (!collection_offsets.count(name)) {
            std::cout << name << ": " << "Unknown" << std::endl;
        }
    }

    return collection_offsets;
}

int discrete_main(
    const cv::Mat& slice_mat,
    const VCCollection& point_collection,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const std::string& umbilicus_set_name,
    int iterations,
    const fs::path& output_path,
    const fs::path& conflicts_path
) {
    // Initialize label image
    cv::Mat labels(slice_mat.size(), CV_32S, cv::Scalar(0));
    const int32_t zero_label = 0;

    std::unordered_map<int32_t, std::string> label_to_name_map;
    std::unordered_map<std::string, int32_t> collection_base_labels;
    int32_t next_base_label = BASE_LABEL_STEP;

    std::unordered_set<cv::Point, PointHash> active_pixels;

    const auto& collections = point_collection.getAllCollections();
    for (const auto& [id, collection] : collections) {
        if (collection.name == umbilicus_set_name) continue;

        if (collection.metadata.absolute_winding_number) {
            std::cout << "Warning: Collection '" << collection.name << "' has absolute winding numbers, which is not fully supported. Processing anyway." << std::endl;
        }

        if (collection_base_labels.find(collection.name) == collection_base_labels.end()) {
            collection_base_labels[collection.name] = next_base_label;
            label_to_name_map[next_base_label] = collection.name;
            next_base_label += BASE_LABEL_STEP;
        }
        int32_t base_label = collection_base_labels.at(collection.name);

        for (const auto& [point_id, point] : collection.points) {
            if (std::isnan(point.winding_annotation)) continue;

            float winding_val = point.winding_annotation;
            if (winding_val != std::round(winding_val)) {
                std::cerr << "Error: Winding annotation for point " << point_id << " in collection '" << collection.name << "' is not an integer (" << winding_val << ")." << std::endl;
                return 1;
            }
            int winding_int = static_cast<int>(std::round(winding_val));

            int x = static_cast<int>(std::round(point.p[0]));
            int y = static_cast<int>(std::round(point.p[1]));

            if (x < 0 || x >= labels.cols || y < 0 || y >= labels.rows) continue;

            int32_t final_label = base_label + winding_int;
            labels.at<int32_t>(y, x) = final_label;
            active_pixels.insert(cv::Point(x, y));
        }
    }
    std::cout << "Initialized labels from point sets." << std::endl;

    // Diffusion loop
    int umb_slice_x = static_cast<int>(std::round((*umbilicus_point)[0]));
    int umb_slice_y = static_cast<int>(std::round((*umbilicus_point)[1]));

    std::vector<std::tuple<cv::Point, int32_t, int32_t>> conflicts;
    cv::Mat labels_curr = labels.clone();

    cv::Mat* labels_prev = &labels;
    cv::Mat* labels_curr_ptr = &labels_curr;

    auto start_diffusion = std::chrono::high_resolution_clock::now();
    auto last_report_time = start_diffusion;
    int last_report_iter = 0;
    for (int i = 0; i < iterations; ++i) {
        if (active_pixels.empty()) {
            std::cout << "No more active pixels. Stopping at iteration " << i << std::endl;
            break;
        }

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_since_last_report = now - last_report_time;
        if (elapsed_since_last_report.count() >= 1.0) {
            int iters_since_last_report = i - last_report_iter;
            std::cout << "Iteration " << i << "/" << iterations << " (" << active_pixels.size() << " active pixels). "
            << iters_since_last_report / elapsed_since_last_report.count() << " iters/s" << std::endl;
            last_report_time = now;
            last_report_iter = i;
        }

        std::unordered_set<cv::Point, PointHash> next_active_pixels;

        for (const auto& p : active_pixels) {
            labels_curr_ptr->at<int32_t>(p.y, p.x) = labels_prev->at<int32_t>(p.y, p.x);
        }

        for (const auto& p : active_pixels) {
            int32_t current_label = labels_prev->at<int32_t>(p.y, p.x);

            int dx[] = {0, 0, 1, -1};
            int dy[] = {1, -1, 0, 0};
            for (int j = 0; j < 4; ++j) {
                int nx = p.x + dx[j];
                int ny = p.y + dy[j];

                if (nx >= 0 && nx < labels.cols && ny >= 0 && ny < labels.rows && slice_mat.at<uint8_t>(ny, nx) != 0) {
                    int32_t propagated_label = current_label;
                    // Check for split line crossing (vertical line down from umbilicus)
                    if (p.y >= umb_slice_y && ny == p.y) { // Horizontal neighbor below or at umbilicus
                        if (p.x < umb_slice_x && nx >= umb_slice_x) { // Crossing from left to right
                            propagated_label++;
                        } else if (p.x >= umb_slice_x && nx < umb_slice_x) { // Crossing from right to left
                            propagated_label--;
                        }
                    }

                    int32_t& target_label = labels_curr_ptr->at<int32_t>(ny, nx);
                    if (target_label == zero_label) {
                        target_label = propagated_label;
                        next_active_pixels.insert(cv::Point(nx, ny));
                    } else if (target_label != propagated_label) {
                        conflicts.emplace_back(cv::Point(nx, ny), target_label, propagated_label);
                    }
                }
            }
        }
        active_pixels = next_active_pixels;
        std::swap(labels_prev, labels_curr_ptr);
    }
    labels = *labels_prev;
    auto end_diffusion = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_diffusion_time = end_diffusion - start_diffusion;
    std::cout << "Diffusion finished after " << total_diffusion_time.count() << " s." << std::endl;

    // Group, sort, and report conflicts
    auto all_conflict_groups = group_and_sort_conflicts(conflicts);
    print_conflict_report(all_conflict_groups, label_to_name_map, {});

    // Estimate and report offsets
    auto collection_offsets = estimate_offsets(all_conflict_groups, label_to_name_map);
    print_conflict_report(all_conflict_groups, label_to_name_map, collection_offsets);

    // Final Visualizations
    visualize_labels(labels, label_to_name_map, collection_offsets, output_path);
    visualize_conflicts(labels, conflicts, label_to_name_map, collection_offsets, conflicts_path);
    return 0;
}


void visualize_labels(
    const cv::Mat& labels,
    const std::unordered_map<int32_t, std::string>& label_to_name_map,
    const std::unordered_map<std::string, int>& collection_offsets,
    const fs::path& output_path
) {
    cv::Mat viz(labels.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    std::unordered_map<int32_t, cv::Vec3b> viz_color_map;
    std::random_device rd_viz;
    std::mt19937 gen_viz(rd_viz());
    std::uniform_int_distribution<> distrib_viz(0, 255);

    for (int y = 0; y < viz.rows; ++y) {
        for (int x = 0; x < viz.cols; ++x) {
            int32_t original_label = labels.at<int32_t>(y, x);
            if (original_label != 0) {
                int32_t final_label = original_label;
                const std::string& name = get_collection_name_from_label(original_label, label_to_name_map);
                if (collection_offsets.count(name)) {
                    final_label += collection_offsets.at(name);
                }

                if (viz_color_map.find(final_label) == viz_color_map.end()) {
                    viz_color_map[final_label] = cv::Vec3b(distrib_viz(gen_viz), distrib_viz(gen_viz), distrib_viz(gen_viz));
                }
                viz.at<cv::Vec3b>(y, x) = viz_color_map[final_label];
            }
        }
    }

    if (!cv::imwrite(output_path.string(), viz)) {
        std::cerr << "Error: Failed to write output image to " << output_path << std::endl;
    } else {
        std::cout << "Saved visualization to " << output_path << std::endl;
    }
}

void visualize_conflicts(
    const cv::Mat& labels,
    const std::vector<std::tuple<cv::Point, int32_t, int32_t>>& conflicts,
    const std::unordered_map<int32_t, std::string>& label_to_name_map,
    const std::unordered_map<std::string, int>& collection_offsets,
    const fs::path& conflicts_path
) {
    if (conflicts_path.empty()) return;

    cv::Mat conflicts_viz(labels.size(), CV_8UC1, cv::Scalar(0));
    for (const auto& conflict : conflicts) {
        const auto& p = std::get<0>(conflict);
        int32_t l1 = std::get<1>(conflict);
        int32_t l2 = std::get<2>(conflict);

        const auto& name1 = get_collection_name_from_label(l1, label_to_name_map);
        const auto& name2 = get_collection_name_from_label(l2, label_to_name_map);

        int32_t final1 = l1 + (collection_offsets.count(name1) ? collection_offsets.at(name1) : 0);
        int32_t final2 = l2 + (collection_offsets.count(name2) ? collection_offsets.at(name2) : 0);

        if (final1 != final2) {
            conflicts_viz.at<uint8_t>(p.y, p.x) = 255; // Unexplained
        } else {
            conflicts_viz.at<uint8_t>(p.y, p.x) = 128; // Explained
        }
    }
    if (!cv::imwrite(conflicts_path.string(), conflicts_viz)) {
        std::cerr << "Error: Failed to write conflicts map to " << conflicts_path << std::endl;
    } else {
        std::cout << "Saved conflicts map to " << conflicts_path << std::endl;
    }
}

std::vector<ConflictGroup> group_and_sort_conflicts(
    const std::vector<std::tuple<cv::Point, int32_t, int32_t>>& conflicts
) {
    std::vector<ConflictGroup> all_conflict_groups;
    if (conflicts.empty()) return all_conflict_groups;

    std::unordered_map<cv::Point, std::vector<std::tuple<cv::Point, int32_t, int32_t>>, PointHash> grouped_conflicts;
    for(const auto& conflict : conflicts) {
        grouped_conflicts[std::get<0>(conflict)].push_back(conflict);
    }

    std::unordered_set<cv::Point, PointHash> visited;
    for (const auto& conflict : conflicts) {
        const auto& p = std::get<0>(conflict);
        if (visited.count(p)) continue;

        std::vector<cv::Point> current_group_points;
        std::vector<cv::Point> stack;
        stack.push_back(p);
        visited.insert(p);

        int32_t group_l1 = std::get<1>(conflict);
        int32_t group_l2 = std::get<2>(conflict);

        while (!stack.empty()) {
            cv::Point current_p = stack.back();
            stack.pop_back();
            current_group_points.push_back(current_p);

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    cv::Point neighbor(current_p.x + dx, current_p.y + dy);
                    if (grouped_conflicts.count(neighbor) && !visited.count(neighbor)) {
                        visited.insert(neighbor);
                        stack.push_back(neighbor);
                    }
                }
            }
        }
        all_conflict_groups.push_back({
            (int)current_group_points.size(),
                                      group_l1,
                                      group_l2,
                                      (int)(group_l2 - group_l1)
        });
    }

    std::sort(all_conflict_groups.begin(), all_conflict_groups.end(), [](const ConflictGroup& a, const ConflictGroup& b) {
        return a.size > b.size;
    });

    return all_conflict_groups;
}


void calculate_sheet_distance_constraints(
    const cv::Mat& slice_mat,
    const cv::Point& umbilicus,
    float ray_step_dist,
    std::vector<SheetConstraintRay>& constraint_rays,
    std::unordered_map<cv::Point, cv::Point, PointHash>& constraints
) {
    int h = slice_mat.rows;
    int w = slice_mat.cols;

    float max_dist = 0;
    max_dist = std::max(max_dist, (float)cv::norm(cv::Point(0, 0) - umbilicus));
    max_dist = std::max(max_dist, (float)cv::norm(cv::Point(w - 1, 0) - umbilicus));
    max_dist = std::max(max_dist, (float)cv::norm(cv::Point(0, h - 1) - umbilicus));
    max_dist = std::max(max_dist, (float)cv::norm(cv::Point(w - 1, h - 1) - umbilicus));

    float min_dist = 0.5f * ray_step_dist;
    int grid_cols = static_cast<int>(std::ceil(w / min_dist));
    int grid_rows = static_cast<int>(std::ceil(h / min_dist));
    std::vector<std::vector<std::vector<cv::Point2f>>> grid(grid_rows, std::vector<std::vector<cv::Point2f>>(grid_cols));

    for (float angle = 0; angle < 2 * CV_PI; angle += ray_step_dist / (max_dist + 1e-6f)) {
        cv::Point2f dir(std::cos(angle), std::sin(angle));

        std::vector<cv::Point> current_segment;
        std::vector<cv::Point> segment_centers;

        for (float dist = max_dist; dist >= -1; --dist) {
            cv::Point p;
            bool is_on_sheet = false;

            if (dist >= 0) {
                cv::Point2f p_float = cv::Point2f(umbilicus.x, umbilicus.y) + dir * dist;
                p = cv::Point(static_cast<int>(std::round(p_float.x)), static_cast<int>(std::round(p_float.y)));
                if (p.x >= 0 && p.x < w && p.y >= 0 && p.y < h) {
                    is_on_sheet = slice_mat.at<uint8_t>(p) > 0;
                }
            }

            if (is_on_sheet) {
                current_segment.push_back(p);
            } else {
                if (!current_segment.empty()) {
                    cv::Point2f avg(0, 0);
                    for(const auto& pt : current_segment) avg += cv::Point2f(pt.x, pt.y);
                    avg *= (1.0f / current_segment.size());
                    segment_centers.emplace_back(cv::Point(avg.x, avg.y));
                    current_segment.clear();
                }
            }
        }

        if (segment_centers.size() > 1) {
            SheetConstraintRay current_ray;
            current_ray.dir = dir;
            for (size_t i = 0; i < segment_centers.size() - 1; ++i) {
                const auto& p1 = segment_centers[i];
                const auto& p2 = segment_centers[i+1];

                bool too_close = false;
                int grid_x = static_cast<int>(p1.x / min_dist);
                int grid_y = static_cast<int>(p1.y / min_dist);

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ny = grid_y + dy;
                        int nx = grid_x + dx;
                        if (ny >= 0 && ny < grid_rows && nx >= 0 && nx < grid_cols) {
                            for (const auto& center : grid[ny][nx]) {
                                if (cv::norm(cv::Point2f(p1.x, p1.y) - center) < min_dist) {
                                    too_close = true;
                                    break;
                                }
                            }
                        }
                        if (too_close) break;
                    }
                    if (too_close) break;
                }

                if (!too_close) {
                    constraints[p1] = p2;
                    current_ray.constraints.push_back({p1, p2});
                    if (grid_y >= 0 && grid_y < grid_rows && grid_x >= 0 && grid_x < grid_cols) {
                        grid[grid_y][grid_x].push_back(cv::Point2f(p1.x, p1.y));
                    }
                }
            }
            if (!current_ray.constraints.empty()) {
                constraint_rays.push_back(current_ray);
            }
        }
    }
    std::cout << "Generated " << constraints.size() << " sheet distance constraints across " << constraint_rays.size() << " rays." << std::endl;
}
