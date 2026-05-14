// Implementation of the merge.json grid parser and connectivity check,
// factored out of vc_merge_tifxyz.cpp for testability. No behavior change
// from the previous in-place definitions.

#include "vc_merge_tifxyz_grid.hpp"

#include <nlohmann/json.hpp>

#include <deque>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vc::merge {

namespace fs = std::filesystem;
using json = nlohmann::json;

void gmResolveGrid(const fs::path& merge_path,
                   const fs::path& paths_dir,
                   std::vector<GMSurfaceSpec>& surfaces,
                   std::vector<GMEdgeSpec>& edges)
{
    const fs::path& mj = merge_path;
    std::ifstream f(mj);
    if (!f) throw std::runtime_error("cannot open " + mj.string());
    json j; f >> j;

    if (j.size() != 1 || !j.contains("rows"))
        throw std::runtime_error(mj.string() +
            ": only the 'rows' key is accepted");
    const auto& rows_j = j.at("rows");
    if (!rows_j.is_array() || rows_j.empty())
        throw std::runtime_error(mj.string() + ": 'rows' must be a non-empty array");

    const size_t R = rows_j.size();
    std::vector<std::vector<std::string>> grid(R);
    std::unordered_map<std::string, fs::path> name_to_path;

    // Each row may be either a JSON array of strings (legacy) OR a single
    // whitespace-delimited string. The string form is easier to hand-edit:
    //   "rows": ["name_a name_b name_c", "name_d name_e name_f"]
    auto splitWhitespace = [](const std::string& s) {
        std::vector<std::string> out;
        std::istringstream iss(s);
        std::string tok;
        while (iss >> tok) out.push_back(std::move(tok));
        return out;
    };

    for (size_t r = 0; r < R; ++r) {
        const auto& row_j = rows_j[r];
        std::vector<std::string> row_names;
        if (row_j.is_array()) {
            row_names.reserve(row_j.size());
            for (size_t c = 0; c < row_j.size(); ++c) {
                const auto& cell = row_j[c];
                if (cell.is_null()) { row_names.emplace_back(); continue; }
                if (!cell.is_string())
                    throw std::runtime_error(mj.string() + ": rows[" +
                        std::to_string(r) + "][" + std::to_string(c) +
                        "] must be a string or null");
                row_names.push_back(cell.get<std::string>());
            }
        } else if (row_j.is_string()) {
            row_names = splitWhitespace(row_j.get<std::string>());
        } else {
            throw std::runtime_error(mj.string() + ": rows[" + std::to_string(r) +
                "] must be an array or a whitespace-delimited string");
        }

        const size_t C = row_names.size();
        grid[r].resize(C);
        for (size_t c = 0; c < C; ++c) {
            const std::string& name = row_names[c];
            if (name.empty()) continue;
            if (name_to_path.count(name)) {
                grid[r][c] = name;  // duplicate cell -> same surface
                continue;
            }
            const fs::path dir = paths_dir / name;
            if (!fs::is_directory(dir))
                throw std::runtime_error(mj.string() + ": rows[" +
                    std::to_string(r) + "][" + std::to_string(c) + "] = '" +
                    name + "' is not a directory under " + paths_dir.string());
            name_to_path[name] = dir;
            grid[r][c] = name;
        }
    }

    surfaces.clear();
    surfaces.reserve(name_to_path.size());
    std::set<std::string> seen;
    for (size_t r = 0; r < R; ++r) {
        for (size_t c = 0; c < grid[r].size(); ++c) {
            const std::string& n = grid[r][c];
            if (n.empty() || !seen.insert(n).second) continue;
            GMSurfaceSpec s;
            s.name = n;
            s.path = name_to_path.at(n);
            surfaces.push_back(std::move(s));
        }
    }

    if (surfaces.size() < 2)
        throw std::runtime_error(mj.string() + ": need at least 2 distinct "
            "surfaces in the grid; got " + std::to_string(surfaces.size()));

    auto edgeKey = [](const std::string& a, const std::string& b) {
        return a < b ? a + "\t" + b : b + "\t" + a;
    };
    std::set<std::string> edge_seen;
    edges.clear();
    auto addEdge = [&](const std::string& a, const std::string& b) {
        if (a.empty() || b.empty() || a == b) return;
        if (edge_seen.insert(edgeKey(a, b)).second) edges.push_back({a, b});
    };
    for (size_t r = 0; r < R; ++r) {
        const auto& row = grid[r];
        for (size_t c = 0; c + 1 < row.size(); ++c) addEdge(row[c], row[c + 1]);
    }
    for (size_t r = 0; r + 1 < R; ++r) {
        const size_t C = std::min(grid[r].size(), grid[r + 1].size());
        for (size_t c = 0; c < C; ++c) addEdge(grid[r][c], grid[r + 1][c]);
    }
}

void gmCheckConnected(const std::vector<GMSurfaceSpec>& surfaces,
                      const std::vector<GMEdgeSpec>& edges)
{
    std::unordered_map<std::string, std::vector<std::string>> adj;
    for (const auto& s : surfaces) adj[s.name];
    for (const auto& e : edges) {
        adj[e.a].push_back(e.b);
        adj[e.b].push_back(e.a);
    }
    std::unordered_set<std::string> visited;
    std::deque<std::string> q;
    q.push_back(surfaces.front().name);
    visited.insert(surfaces.front().name);
    while (!q.empty()) {
        const std::string u = q.front(); q.pop_front();
        for (const auto& v : adj[u])
            if (visited.insert(v).second) q.push_back(v);
    }
    if (visited.size() == surfaces.size()) return;
    std::ostringstream msg;
    msg << "edge graph from merge.json is disconnected; "
           "unreachable surfaces:";
    for (const auto& s : surfaces)
        if (!visited.count(s.name)) msg << "\n  " << s.name;
    throw std::runtime_error(msg.str());
}

} // namespace vc::merge
