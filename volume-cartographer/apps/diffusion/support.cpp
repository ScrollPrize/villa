#include "support.hpp"
#include "spiral_ceres.hpp"
#include "vc/core/util/LifeTime.hpp"

#include <vc/core/util/GridStore.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>

#include <opencv2/ximgproc.hpp>

void visualize_normal_grid(const vc::core::util::GridStore& normal_grid, const cv::Size& size, const std::string& path);



void populate_normal_grid(const SkeletonGraph& g, vc::core::util::GridStore& normal_grid, double spiral_step) {
    int path_count = 0;
    const float target_length = static_cast<float>(spiral_step);

    for (const auto& edge : boost::make_iterator_range(boost::edges(g))) {
        const auto& path = g[edge].path;
        if (path.size() < 2) continue;

        std::vector<cv::Point> resampled_path;
        resampled_path.push_back(path[0]);

        float accumulated_dist = 0.f;
        cv::Point current_pos = path[0];

        for (size_t i = 0; i < path.size() - 1; ++i) {
            cv::Point p1 = path[i];
            cv::Point p2 = path[i+1];
            float segment_length = cv::norm(p2 - p1);

            while (accumulated_dist + segment_length >= target_length) {
                float remaining_dist = target_length - accumulated_dist;
                cv::Point2f vec = cv::Point2f(p2 - p1) / segment_length;
                current_pos = cv::Point(cv::Point2f(p1) + vec * remaining_dist);
                resampled_path.push_back(current_pos);

                p1 = current_pos;
                segment_length -= remaining_dist;
                accumulated_dist = 0;
            }
            accumulated_dist += segment_length;
        }

        if (resampled_path.size() >= 2) {
            normal_grid.add(resampled_path);
            path_count++;
        }
    }
}


void visualize_normal_grid(const vc::core::util::GridStore& normal_grid, const cv::Size& size, const std::string& path) {
    cv::Mat normal_constraints_vis = cv::Mat::zeros(size, CV_8UC3);
    cv::RNG rng(12345);
    const auto& all_paths = normal_grid.get_all();
    std::cout << "Visualizing " << all_paths.size() << " paths from the grid store." << std::endl;
    for (const auto& path_ptr : all_paths) {
        const auto& path = *path_ptr;
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        for (size_t i = 0; i < path.size() - 1; ++i) {
            cv::line(normal_constraints_vis, path[i], path[i+1], color, 1);
            cv::circle(normal_constraints_vis, path[i+1], 3, color, -1);
        }
    }
    cv::imwrite(path, normal_constraints_vis);
}


std::pair<SkeletonGraph, cv::Mat> generate_skeleton_graph(const cv::Mat& binary_slice, const po::variables_map& vm) {
    cv::Mat skeleton;
    cv::ximgproc::thinning(binary_slice, skeleton, cv::ximgproc::THINNING_GUOHALL);

    SkeletonGraph g;
    cv::Mat skeleton_id_img = cv::Mat::zeros(skeleton.size(), CV_32S);
    skeleton_id_img.setTo(-1);
    std::unordered_map<cv::Point, int, PointHash> vertex_map;
    int next_edge_id = 1;

    // 1. Find Junctions and Endpoints
    for (int y = 0; y < skeleton.rows; ++y) {
        for (int x = 0; x < skeleton.cols; ++x) {
            if (skeleton.at<uint8_t>(y, x) == 0) continue;

            int neighbors = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && ny < skeleton.rows && nx >= 0 && nx < skeleton.cols && skeleton.at<uint8_t>(ny, nx) > 0) {
                        neighbors++;
                    }
                }
            }

            if (neighbors != 2) {
                cv::Point p(x, y);
                if (vertex_map.find(p) == vertex_map.end()) {
                    int v_id = boost::add_vertex(g);
                    g[v_id].pos = p;
                    vertex_map[p] = v_id;
                }
            }
        }
    }

    // 2. Trace Edges and 3. Populate skeleton_id_img
    cv::Mat visited = cv::Mat::zeros(skeleton.size(), CV_8U);
    auto trace_from_vertex = [&](const cv::Point& p, int v_id) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                cv::Point neighbor(p.x + dx, p.y + dy);
                if (neighbor.x >= 0 && neighbor.x < skeleton.cols && neighbor.y >= 0 && neighbor.y < skeleton.rows &&
                    skeleton.at<uint8_t>(neighbor) > 0 && visited.at<uint8_t>(neighbor) == 0) {

                    std::vector<cv::Point> path;
                path.push_back(p);
                visited.at<uint8_t>(p) = 1;

                cv::Point current_p = neighbor;
                cv::Point prev_p = p;

                bool path_terminated = false;
                while (vertex_map.find(current_p) == vertex_map.end()) {
                    path.push_back(current_p);
                    visited.at<uint8_t>(current_p) = 1;

                    bool found_next = false;
                    for (int ndy = -1; ndy <= 1; ++ndy) {
                        for (int ndx = -1; ndx <= 1; ++ndx) {
                            if (ndx == 0 && ndy == 0) continue;
                            cv::Point next_p(current_p.x + ndx, current_p.y + ndy);
                            if (next_p != prev_p &&
                                next_p.x >= 0 && next_p.x < skeleton.cols && next_p.y >= 0 && next_p.y < skeleton.rows &&
                                skeleton.at<uint8_t>(next_p) > 0) {
                                prev_p = current_p;
                            current_p = next_p;
                            found_next = true;
                            break;
                                }
                        }
                        if (found_next) break;
                    }
                    if (!found_next) {
                        path_terminated = true;
                        break;
                    }
                }

                int other_v_id;
                bool add_edge = false;
                if (path_terminated) {
                    other_v_id = boost::add_vertex(g);
                    g[other_v_id].pos = path.back();
                    vertex_map[path.back()] = other_v_id;
                    add_edge = true;
                } else {
                    path.push_back(current_p);
                    visited.at<uint8_t>(current_p) = 1;
                    other_v_id = vertex_map.at(current_p);
                    add_edge = !boost::edge(v_id, other_v_id, g).second;
                }

                if (add_edge) {
                    auto edge_desc = boost::add_edge(v_id, other_v_id, g).first;
                    g[edge_desc].path = path;
                    g[edge_desc].id = next_edge_id;
                    for(const auto& path_p : path) {
                        skeleton_id_img.at<int32_t>(path_p) = next_edge_id;
                    }
                    next_edge_id++;
                }
                    }
            }
        }
    };

    for (auto const& [p, v_id] : vertex_map) {
        trace_from_vertex(p, v_id);
    }

    // 4. Second pass for missed segments (e.g., loops)
    for (int y = 0; y < skeleton.rows; ++y) {
        for (int x = 0; x < skeleton.cols; ++x) {
            if (skeleton.at<uint8_t>(y, x) > 0 && visited.at<uint8_t>(y, x) == 0) {
                cv::Point start_p(x, y);

                // Create two new vertices for the loop
                int v1_id = boost::add_vertex(g);
                g[v1_id].pos = start_p;
                vertex_map[start_p] = v1_id;

                std::vector<cv::Point> path;
                path.push_back(start_p);
                visited.at<uint8_t>(start_p) = 1;

                cv::Point current_p = start_p;
                cv::Point prev_p = start_p;

                // Find first step
                bool found_first = false;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        cv::Point next_p(current_p.x + dx, current_p.y + dy);
                        if (next_p.x >= 0 && next_p.x < skeleton.cols && next_p.y >= 0 && next_p.y < skeleton.rows &&
                            skeleton.at<uint8_t>(next_p) > 0) {
                            current_p = next_p;
                        found_first = true;
                        break;
                            }
                    }
                    if(found_first) break;
                }

                while (current_p != start_p) {
                    path.push_back(current_p);
                    visited.at<uint8_t>(current_p) = 1;
                    bool found_next = false;
                    for (int ndy = -1; ndy <= 1; ++ndy) {
                        for (int ndx = -1; ndx <= 1; ++ndx) {
                            if (ndx == 0 && ndy == 0) continue;
                            cv::Point next_p(current_p.x + ndx, current_p.y + ndy);
                            if (next_p != prev_p &&
                                next_p.x >= 0 && next_p.x < skeleton.cols && next_p.y >= 0 && next_p.y < skeleton.rows &&
                                skeleton.at<uint8_t>(next_p) > 0) {
                                prev_p = current_p;
                            current_p = next_p;
                            found_next = true;
                            break;
                                }
                        }
                        if (found_next) break;
                    }
                    if (!found_next) break;
                }
                path.push_back(start_p);

                int v2_id = boost::add_vertex(g);
                g[v2_id].pos = start_p;

                auto edge_desc = boost::add_edge(v1_id, v2_id, g);
                g[edge_desc.first].path = path;
                g[edge_desc.first].id = next_edge_id;
                for(const auto& path_p : path) {
                    skeleton_id_img.at<int32_t>(path_p) = next_edge_id;
                }
                next_edge_id++;
            }
        }
    }

    if (vm.count("debug")) {
        cv::Mat vertex_viz;
        cv::cvtColor(skeleton, vertex_viz, cv::COLOR_GRAY2BGR);
        for (const auto& [p, v_id] : vertex_map) {
            cv::circle(vertex_viz, p, 3, cv::Scalar(0, 0, 255), -1);
        }
        cv::imwrite("skeleton_vertices.tif", vertex_viz);
    }

    if (vm.count("debug")) {
        cv::Mat vertex_viz;
        cv::cvtColor(skeleton, vertex_viz, cv::COLOR_GRAY2BGR);
        for (const auto& [p, v_id] : vertex_map) {
            cv::circle(vertex_viz, p, 3, cv::Scalar(0, 0, 255), -1);
        }
        cv::imwrite("skeleton_vertices.tif", vertex_viz);
    }

    return {g, skeleton_id_img};
}
