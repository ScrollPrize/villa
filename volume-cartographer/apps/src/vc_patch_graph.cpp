#include "vc/core/util/QuadSurface.hpp"

#include <boost/program_options.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

enum class GraphMode {
    Vertices,
    Quads,
};

struct GridCoord {
    int row = -1;
    int col = -1;
};

struct Graph {
    int rows = 0;
    int cols = 0;
    std::vector<int> nodeIds;
    std::vector<GridCoord> coords;
    std::vector<cv::Vec3f> positions;
};

struct QueueItem {
    double dist = 0.0;
    int node = -1;

    bool operator>(const QueueItem& other) const
    {
        return dist > other.dist;
    }
};

GridCoord parseGridCoord(const std::string& text)
{
    const auto comma = text.find(',');
    if (comma == std::string::npos) {
        throw std::runtime_error("expected grid coordinate as row,col: " + text);
    }

    GridCoord coord;
    std::size_t rowPos = 0;
    std::size_t colPos = 0;
    coord.row = std::stoi(text.substr(0, comma), &rowPos);
    coord.col = std::stoi(text.substr(comma + 1), &colPos);
    if (rowPos != comma || colPos != text.size() - comma - 1) {
        throw std::runtime_error("invalid grid coordinate: " + text);
    }
    return coord;
}

int linearIndex(int row, int col, int cols)
{
    return row * cols + col;
}

double distance3d(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const cv::Vec3d d{
        static_cast<double>(a[0]) - static_cast<double>(b[0]),
        static_cast<double>(a[1]) - static_cast<double>(b[1]),
        static_cast<double>(a[2]) - static_cast<double>(b[2]),
    };
    return cv::norm(d);
}

cv::Vec3f quadCenter(const cv::Mat_<cv::Vec3f>& points, int row, int col)
{
    return (points(row, col) +
            points(row, col + 1) +
            points(row + 1, col) +
            points(row + 1, col + 1)) * 0.25f;
}

Graph buildVertexGraph(const QuadSurface& surface)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    Graph graph;
    graph.rows = points->rows;
    graph.cols = points->cols;
    graph.nodeIds.assign(static_cast<std::size_t>(graph.rows) * graph.cols, -1);

    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            if (!surface.isPointValid(row, col)) {
                continue;
            }
            const int id = static_cast<int>(graph.coords.size());
            graph.nodeIds[linearIndex(row, col, graph.cols)] = id;
            graph.coords.push_back({row, col});
            graph.positions.push_back((*points)(row, col));
        }
    }

    return graph;
}

Graph buildQuadGraph(const QuadSurface& surface)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    Graph graph;
    graph.rows = std::max(0, points->rows - 1);
    graph.cols = std::max(0, points->cols - 1);
    graph.nodeIds.assign(static_cast<std::size_t>(graph.rows) * graph.cols, -1);

    for (int row = 0; row < graph.rows; ++row) {
        for (int col = 0; col < graph.cols; ++col) {
            if (!surface.isQuadValid(row, col)) {
                continue;
            }
            const int id = static_cast<int>(graph.coords.size());
            graph.nodeIds[linearIndex(row, col, graph.cols)] = id;
            graph.coords.push_back({row, col});
            graph.positions.push_back(quadCenter(*points, row, col));
        }
    }

    return graph;
}

int nodeAt(const Graph& graph, GridCoord coord, const char* label)
{
    if (coord.row < 0 || coord.row >= graph.rows ||
        coord.col < 0 || coord.col >= graph.cols) {
        std::ostringstream oss;
        oss << label << " coordinate out of range: " << coord.row << "," << coord.col
            << " for graph dimensions " << graph.rows << "x" << graph.cols;
        throw std::runtime_error(oss.str());
    }

    const int node = graph.nodeIds[linearIndex(coord.row, coord.col, graph.cols)];
    if (node < 0) {
        std::ostringstream oss;
        oss << label << " coordinate is not a valid graph node: "
            << coord.row << "," << coord.col;
        throw std::runtime_error(oss.str());
    }
    return node;
}

template <typename Fn>
void forEachNeighbor(const Graph& graph, int node, Fn&& fn)
{
    static constexpr int kDRow[] = {-1, 0, 1, 0};
    static constexpr int kDCol[] = {0, 1, 0, -1};

    const GridCoord coord = graph.coords[node];
    for (int i = 0; i < 4; ++i) {
        const int row = coord.row + kDRow[i];
        const int col = coord.col + kDCol[i];
        if (row < 0 || row >= graph.rows || col < 0 || col >= graph.cols) {
            continue;
        }
        const int neighbor = graph.nodeIds[linearIndex(row, col, graph.cols)];
        if (neighbor >= 0) {
            fn(neighbor);
        }
    }
}

std::vector<int> shortestPath(const Graph& graph, int start, int goal, double* outDistance)
{
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dist(graph.coords.size(), inf);
    std::vector<int> parent(graph.coords.size(), -1);
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;

    dist[start] = 0.0;
    queue.push({0.0, start});

    while (!queue.empty()) {
        const QueueItem current = queue.top();
        queue.pop();
        if (current.dist != dist[current.node]) {
            continue;
        }
        if (current.node == goal) {
            break;
        }

        forEachNeighbor(graph, current.node, [&](int neighbor) {
            const double edge = distance3d(graph.positions[current.node], graph.positions[neighbor]);
            const double candidate = current.dist + edge;
            if (candidate < dist[neighbor]) {
                dist[neighbor] = candidate;
                parent[neighbor] = current.node;
                queue.push({candidate, neighbor});
            }
        });
    }

    *outDistance = dist[goal];
    if (!std::isfinite(dist[goal])) {
        return {};
    }

    std::vector<int> path;
    for (int node = goal; node >= 0; node = parent[node]) {
        path.push_back(node);
        if (node == start) {
            break;
        }
    }
    std::reverse(path.begin(), path.end());
    return path;
}

void writePathCsv(const Graph& graph, const std::vector<int>& path, const fs::path& output)
{
    std::ostream* stream = &std::cout;
    std::ofstream file;
    if (!output.empty()) {
        file.open(output);
        if (!file) {
            throw std::runtime_error("failed to open output path: " + output.string());
        }
        stream = &file;
    }

    *stream << "step,node,row,col,x,y,z,cumulative_distance\n";
    double cumulative = 0.0;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const int node = path[i];
        if (i > 0) {
            cumulative += distance3d(graph.positions[path[i - 1]], graph.positions[node]);
        }
        const GridCoord coord = graph.coords[node];
        const cv::Vec3f pos = graph.positions[node];
        *stream << i << ','
                << node << ','
                << coord.row << ','
                << coord.col << ','
                << std::setprecision(9) << pos[0] << ','
                << std::setprecision(9) << pos[1] << ','
                << std::setprecision(9) << pos[2] << ','
                << std::setprecision(17) << cumulative << '\n';
    }
}

void writeEdgeCsv(const Graph& graph, const fs::path& output)
{
    std::ostream* stream = &std::cout;
    std::ofstream file;
    if (!output.empty()) {
        file.open(output);
        if (!file) {
            throw std::runtime_error("failed to open graph output path: " + output.string());
        }
        stream = &file;
    }

    *stream << "from_node,from_row,from_col,to_node,to_row,to_col,weight\n";
    for (int node = 0; node < static_cast<int>(graph.coords.size()); ++node) {
        forEachNeighbor(graph, node, [&](int neighbor) {
            if (neighbor <= node) {
                return;
            }
            const GridCoord from = graph.coords[node];
            const GridCoord to = graph.coords[neighbor];
            *stream << node << ','
                    << from.row << ','
                    << from.col << ','
                    << neighbor << ','
                    << to.row << ','
                    << to.col << ','
                    << std::setprecision(17)
                    << distance3d(graph.positions[node], graph.positions[neighbor]) << '\n';
        });
    }
}

void printUsage(const char* argv0, const po::options_description& desc)
{
    std::cout
        << "Usage:\n"
        << "  " << argv0 << " <tifxyz> --mode vertices --start row,col --end row,col [--output path.csv]\n"
        << "  " << argv0 << " <tifxyz> --mode quads --start row,col --end row,col [--output path.csv]\n"
        << "  " << argv0 << " <tifxyz> --mode vertices --dump-graph edges.csv\n\n"
        << desc << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input", po::value<std::string>(), "input tifxyz folder")
        ("mode,m", po::value<std::string>()->default_value("vertices"),
         "graph nodes: vertices or quads")
        ("start", po::value<std::string>(), "start grid coordinate as row,col")
        ("end", po::value<std::string>(), "end grid coordinate as row,col")
        ("output,o", po::value<std::string>(), "write shortest path CSV to this file")
        ("dump-graph", po::value<std::string>()->implicit_value(""),
         "write graph edge list CSV instead of a shortest path; use '-' for stdout");

    po::positional_options_description positional;
    positional.add("input", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(desc)
                      .positional(positional)
                      .run(),
                  vm);

        if (vm.count("help")) {
            printUsage(argv[0], desc);
            return EXIT_SUCCESS;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "ERROR: " << e.what() << "\n\n";
        printUsage(argv[0], desc);
        return EXIT_FAILURE;
    }

    try {
        if (!vm.count("input")) {
            throw std::runtime_error("missing input tifxyz folder");
        }

        GraphMode mode;
        const std::string modeText = vm["mode"].as<std::string>();
        if (modeText == "vertices" || modeText == "vertex") {
            mode = GraphMode::Vertices;
        } else if (modeText == "quads" || modeText == "quad") {
            mode = GraphMode::Quads;
        } else {
            throw std::runtime_error("unknown mode: " + modeText);
        }

        const fs::path input = vm["input"].as<std::string>();
        std::unique_ptr<QuadSurface> surface = load_quad_from_tifxyz(input.string());
        const Graph graph = (mode == GraphMode::Vertices)
                                ? buildVertexGraph(*surface)
                                : buildQuadGraph(*surface);

        std::cerr << "graph_mode=" << modeText
                  << " rows=" << graph.rows
                  << " cols=" << graph.cols
                  << " nodes=" << graph.coords.size()
                  << '\n';

        if (vm.count("dump-graph")) {
            fs::path output = vm["dump-graph"].as<std::string>();
            if (output == "-") {
                output.clear();
            }
            writeEdgeCsv(graph, output);
            return EXIT_SUCCESS;
        }

        if (!vm.count("start") || !vm.count("end")) {
            throw std::runtime_error("shortest path requires both --start and --end");
        }

        const int start = nodeAt(graph, parseGridCoord(vm["start"].as<std::string>()), "start");
        const int end = nodeAt(graph, parseGridCoord(vm["end"].as<std::string>()), "end");

        double pathDistance = 0.0;
        const std::vector<int> path = shortestPath(graph, start, end, &pathDistance);
        if (path.empty()) {
            std::cerr << "no path found\n";
            return EXIT_FAILURE;
        }

        fs::path output;
        if (vm.count("output")) {
            output = vm["output"].as<std::string>();
        }
        writePathCsv(graph, path, output);

        std::cerr << "path_nodes=" << path.size()
                  << " distance=" << std::setprecision(17) << pathDistance
                  << '\n';
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
