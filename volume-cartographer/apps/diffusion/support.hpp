#pragma once

#include <opencv2/opencv.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <vector>

#include <vc/core/util/GridStore.hpp>

struct SkeletonVertex {
    cv::Point pos;
};

struct SkeletonEdge {
    std::vector<cv::Point> path;
    int id;
};

using SkeletonGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, SkeletonVertex, SkeletonEdge>;

#include "common.hpp"

#include <opencv2/core.hpp>

struct PointHash {
    std::size_t operator()(const cv::Point& p) const {
        auto h1 = std::hash<int>{}(p.x);
        auto h2 = std::hash<int>{}(p.y);
        return h1 ^ (h2 << 1);
    }
};

void parse_center(cv::Vec3f& center, const std::string& center_str);

std::pair<SkeletonGraph, cv::Mat> generate_skeleton_graph(const cv::Mat& binary_slice, const po::variables_map& vm);
void populate_normal_grid(const SkeletonGraph& graph, vc::core::util::GridStore& normal_grid, double spiral_step);