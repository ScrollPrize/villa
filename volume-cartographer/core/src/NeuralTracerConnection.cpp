#include <iostream>
#include <limits>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <nlohmann/json.hpp>

#include "vc/tracer/NeuralTracerConnection.h"


namespace
{
    float get_float_or_nan(const nlohmann::json& j) {
        return j.is_null() ? std::numeric_limits<float>::quiet_NaN() : j.get<float>();
    }

    std::vector<cv::Vec3f> process_candidates(const nlohmann::json& batch) {
        std::vector<cv::Vec3f> candidates;
        assert(batch.is_array());
        for (auto const& candidate : batch) {
            assert(candidate.is_array() && candidate.size() == 3);
            candidates.emplace_back(
                get_float_or_nan(candidate[0]),
                get_float_or_nan(candidate[1]),
                get_float_or_nan(candidate[2])
            );
        }
        return candidates;
    }

    nlohmann::json process_json_request(const nlohmann::json& req, int sock) {
        std::string response_str;

#pragma omp critical
        {
            std::string msg = req.dump() + "\n";
            if (send(sock, msg.c_str(), msg.length(), 0) < 0)
                throw std::runtime_error("Failed to send request");

            char buffer[1];
            while (recv(sock, buffer, 1, 0) == 1 && buffer[0] != '\n')
                response_str += buffer[0];
        }

        // Replace NaN with null for valid JSON parsing
        size_t pos = 0;
        while ((pos = response_str.find("NaN", pos)) != std::string::npos) {
            bool prefix_ok = (pos == 0) || (response_str[pos-1] == ' ') || (response_str[pos-1] == ',') || (response_str[pos-1] == '[');
            bool suffix_ok = (pos + 3 >= response_str.length()) || (response_str[pos+3] == ' ') || (response_str[pos+3] == ',') || (response_str[pos+3] == ']');
            if (prefix_ok && suffix_ok) {
                response_str.replace(pos, 3, "null");
            }
            pos += 4; // move past "null"
        }

        return nlohmann::json::parse(response_str);
    }
}


NeuralTracerConnection::NeuralTracerConnection(std::string const & socket_path) {

    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    sockaddr_un addr{AF_UNIX};
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        throw std::runtime_error("Socket path too long");
    }
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock, (sockaddr *)&addr, sizeof(addr)) < 0) {
        if (sock >= 0) {
            close(sock);
        }
        throw std::runtime_error("Failed to connect to socket " + socket_path);
    }
}

NeuralTracerConnection::~NeuralTracerConnection() {
    if (sock >= 0) {
        close(sock);
    }
}

std::vector<NeuralTracerConnection::NextUvs> NeuralTracerConnection::get_next_points(
    std::vector<cv::Vec3f> const &center,
    std::vector<std::optional<cv::Vec3f>> const &prev_u,
    std::vector<std::optional<cv::Vec3f>> const &prev_v,
    std::vector<std::optional<cv::Vec3f>> const &prev_diag
) const {
    nlohmann::json req;

    nlohmann::json center_list = nlohmann::json::array();
    for (const auto& c : center) {
        center_list.push_back({c[0], c[1], c[2]});
    }
    req["center_xyz"] = center_list;

    auto convert_prev_coords = [](const std::vector<std::optional<cv::Vec3f>>& coords) {
        nlohmann::json list = nlohmann::json::array();
        for (const auto& p : coords) {
            if (p.has_value()) {
                list.push_back({p.value()[0], p.value()[1], p.value()[2]});
            } else {
                list.push_back(nullptr);
            }
        }
        return list;
    };
    req["prev_u_xyz"] = convert_prev_coords(prev_u);
    req["prev_v_xyz"] = convert_prev_coords(prev_v);
    req["prev_diag_xyz"] = convert_prev_coords(prev_diag);

    nlohmann::json response = process_json_request(req, sock);

    if (response.contains("error")) {
        throw std::runtime_error("Neural tracer returned error: " + response["error"].get<std::string>());
    }

    std::vector<NextUvs> results;
    if (response.contains("u_candidates") && response.contains("v_candidates")) {
        auto& u_batch = response["u_candidates"];
        auto& v_batch = response["v_candidates"];
        assert(u_batch.size() == v_batch.size());
        for (size_t i = 0; i < u_batch.size(); ++i) {
            results.emplace_back(
                process_candidates(u_batch[i]),
                process_candidates(v_batch[i])
            );
        }
    }

    return results;
}

std::vector<NeuralTracerConnection::NextUvsWithJacobian> NeuralTracerConnection::get_next_points_with_jacobian(
    std::vector<cv::Vec3f> const &center,
    std::vector<std::optional<cv::Vec3f>> const &prev_u,
    std::vector<std::optional<cv::Vec3f>> const &prev_v,
    std::vector<std::optional<cv::Vec3f>> const &prev_diag
) const {
    nlohmann::json req;

    nlohmann::json center_list = nlohmann::json::array();
    for (const auto& c : center) {
        center_list.push_back({c[0], c[1], c[2]});
    }
    req["center_xyz"] = center_list;

    auto convert_prev_coords = [](const std::vector<std::optional<cv::Vec3f>>& coords) {
        nlohmann::json list = nlohmann::json::array();
        for (const auto& p : coords) {
            if (p.has_value()) {
                list.push_back({p.value()[0], p.value()[1], p.value()[2]});
            } else {
                list.push_back(nullptr);
            }
        }
        return list;
    };
    req["prev_u_xyz"] = convert_prev_coords(prev_u);
    req["prev_v_xyz"] = convert_prev_coords(prev_v);
    req["prev_diag_xyz"] = convert_prev_coords(prev_diag);
    req["return_jacobian"] = true;

    nlohmann::json response = process_json_request(req, sock);

    if (response.contains("error")) {
        throw std::runtime_error("Neural tracer returned error: " + response["error"].get<std::string>());
    }

    // Parse jacobians arranged by index (center, prev_u, prev_v, prev_diag).
    // Python returns a list for each batch item: [jac_center, jac_prev_u, jac_prev_v, jac_prev_diag],
    // with each jacobian a 3x3 matrix (or a list per candidate if that ever changes) or null.
    auto parse_jacobian_matrix = [&](
        const nlohmann::json& matrix,
        size_t num_candidates
    ) {
        std::vector<std::optional<cv::Matx33f>> result(num_candidates, std::nullopt);

        auto parse_mat3x3 = [&](const nlohmann::json& mat2d) -> std::optional<cv::Matx33f> {
            if (!mat2d.is_array() || mat2d.size() < 3) {
                return std::nullopt;
            }
            for (size_t row = 0; row < 3; ++row) {
                if (!mat2d[row].is_array() || mat2d[row].size() < 3) {
                    return std::nullopt;
                }
            }
            return cv::Matx33f(
                get_float_or_nan(mat2d[0][0]), get_float_or_nan(mat2d[0][1]), get_float_or_nan(mat2d[0][2]),
                get_float_or_nan(mat2d[1][0]), get_float_or_nan(mat2d[1][1]), get_float_or_nan(mat2d[1][2]),
                get_float_or_nan(mat2d[2][0]), get_float_or_nan(mat2d[2][1]), get_float_or_nan(mat2d[2][2])
            );
        };

        auto fill_candidate = [&](size_t cand_idx, const nlohmann::json& mat2d) {
            result[cand_idx] = parse_mat3x3(mat2d);
        };

        if (matrix.is_null()) {
            return result;
        }

        // Support both 2D (3x3) and 3D ([candidate][3][3]) layouts.
        if (matrix.is_array() && !matrix.empty() && matrix[0].is_array() && (matrix[0].empty() || !matrix[0][0].is_array())) {
            // 2D matrix: apply to first candidate (current service returns only one).
            fill_candidate(0, matrix);
        } else if (matrix.is_array()) {
            size_t cand_count = std::min(num_candidates, matrix.size());
            for (size_t cand = 0; cand < cand_count; ++cand) {
                fill_candidate(cand, matrix[cand]);
            }
        }

        return result;
    };

    auto parse_input_jacobians_by_index = [&](
        const nlohmann::json& jac_entry,
        size_t index,
        size_t num_candidates
    ) {
        if (!jac_entry.is_array() || index >= jac_entry.size()) {
            return std::vector<std::optional<cv::Matx33f>>(num_candidates, std::nullopt);
        }
        return parse_jacobian_matrix(jac_entry[index], num_candidates);
    };

    std::vector<NextUvsWithJacobian> results;
    if (response.contains("u_candidates") && response.contains("v_candidates") &&
        response.contains("u_jacobians") && response.contains("v_jacobians")) {
        auto& u_batch = response["u_candidates"];
        auto& v_batch = response["v_candidates"];
        auto& u_jac_batch = response["u_jacobians"];
        auto& v_jac_batch = response["v_jacobians"];
        assert(u_batch.size() == v_batch.size());
        assert(u_batch.size() == u_jac_batch.size());
        assert(u_batch.size() == v_jac_batch.size());

        for (size_t i = 0; i < u_batch.size(); ++i) {
            NextUvsWithJacobian result;
            result.next_u_xyzs = process_candidates(u_batch[i]);
            result.next_v_xyzs = process_candidates(v_batch[i]);

            const auto& u_jac_entry = u_jac_batch[i];
            const auto& v_jac_entry = v_jac_batch[i];

            result.u_jac_wrt_center = parse_input_jacobians_by_index(u_jac_entry, 0, result.next_u_xyzs.size());
            result.u_jac_wrt_prev_u = parse_input_jacobians_by_index(u_jac_entry, 1, result.next_u_xyzs.size());
            result.u_jac_wrt_prev_v = parse_input_jacobians_by_index(u_jac_entry, 2, result.next_u_xyzs.size());
            result.u_jac_wrt_prev_diag = parse_input_jacobians_by_index(u_jac_entry, 3, result.next_u_xyzs.size());

            result.v_jac_wrt_center = parse_input_jacobians_by_index(v_jac_entry, 0, result.next_v_xyzs.size());
            result.v_jac_wrt_prev_u = parse_input_jacobians_by_index(v_jac_entry, 1, result.next_v_xyzs.size());
            result.v_jac_wrt_prev_v = parse_input_jacobians_by_index(v_jac_entry, 2, result.next_v_xyzs.size());
            result.v_jac_wrt_prev_diag = parse_input_jacobians_by_index(v_jac_entry, 3, result.next_v_xyzs.size());

            results.push_back(result);
        }
    }

    return results;
}
