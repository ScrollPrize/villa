#include "support.hpp"
#include "spiral_ceres.hpp"
#include "vc/core/util/LifeTime.hpp"

#include <vc/core/util/GridStore.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>

#include <opencv2/ximgproc.hpp>




void populate_normal_grid(const SkeletonGraph& g, vc::core::util::GridStore& normal_grid, double spiral_step) {
    const float target_dist_sq = static_cast<float>(spiral_step * spiral_step);

    for (const auto& edge : boost::make_iterator_range(boost::edges(g))) {
        const auto& path = g[edge].path;
        if (path.size() < 2) continue;

        std::vector<cv::Point> resampled_path;
        resampled_path.push_back(path[0]);
        cv::Point last_point = path[0];

        for (size_t i = 1; i < path.size(); ++i) {
            cv::Point current_point = path[i];
            double dist_sq = cv::norm(current_point - last_point) * cv::norm(current_point - last_point);
            if (dist_sq >= target_dist_sq) {
                resampled_path.push_back(current_point);
                last_point = current_point;
            }
        }


        if (resampled_path.size() >= 2) {
            normal_grid.add(resampled_path);
        }
    }
}

void populate_normal_grid(const std::vector<std::vector<cv::Point>>& traces, vc::core::util::GridStore& normal_grid, double spiral_step) {
    const float target_dist_sq = static_cast<float>(spiral_step * spiral_step);

    for (const auto& trace : traces) {
        if (trace.size() < 2) continue;

        std::vector<cv::Point> resampled_path;
        resampled_path.push_back(trace[0]);
        cv::Point last_point = trace[0];

        for (size_t i = 1; i < trace.size(); ++i) {
            cv::Point current_point = trace[i];
            double dist_sq = cv::norm(current_point - last_point) * cv::norm(current_point - last_point);
            if (dist_sq >= target_dist_sq) {
                resampled_path.push_back(current_point);
                last_point = current_point;
            }
        }


        if (resampled_path.size() >= 2) {
            normal_grid.add(resampled_path);
        }
    }
}


cv::Mat visualize_normal_grid(const vc::core::util::GridStore& normal_grid, const cv::Size& size) {
    cv::Mat normal_constraints_vis = cv::Mat::zeros(size, CV_8UC3);
    cv::RNG rng(12345);
    const auto& all_paths = normal_grid.get_all();
    std::cout << "Visualizing " << all_paths.size() << " paths from the grid store." << std::endl;
    for (const auto& path_ptr : all_paths) {
        const auto& path = *path_ptr;
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        for (size_t i = 0; i < path.size() - 1; ++i) {
            const auto& p1 = path[i];
            const auto& p2 = path[i+1];

            cv::line(normal_constraints_vis, p1, p2, color, 1);
            cv::circle(normal_constraints_vis, p2, 3, color, -1);

            cv::Point center = (p1 + p2) / 2;
            cv::Vec2f tangent((float)(p2.x - p1.x), (float)(p2.y - p1.y));
            cv::normalize(tangent, tangent);
            cv::Vec2f normal(-tangent[1], tangent[0]);

            cv::Point normal_endpoint(center.x + normal[0] * 5, center.y + normal[1] * 5);
            cv::line(normal_constraints_vis, center, normal_endpoint, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }
    return normal_constraints_vis;
}


SkeletonGraph trace_skeleton_segments(const cv::Mat& skeleton, const po::variables_map& vm) {
    SkeletonGraph g;
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

    return g;
}
