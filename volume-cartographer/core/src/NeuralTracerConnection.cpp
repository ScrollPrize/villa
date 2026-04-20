#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include "utils/Json.hpp"

#include "vc/tracer/NeuralTracerConnection.h"


namespace
{
    utils::Json process_json_request(const utils::Json& req, int sock) {
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

        return utils::Json::parse(response_str);
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
    utils::Json req;

    utils::Json center_list = utils::Json::array();
    for (const auto& c : center) {
        utils::Json pt = utils::Json::array();
        pt.push_back(c[0]); pt.push_back(c[1]); pt.push_back(c[2]);
        center_list.push_back(std::move(pt));
    }
    req["center_xyz"] = center_list;

    auto convert_prev_coords = [](const std::vector<std::optional<cv::Vec3f>>& coords) {
        utils::Json list = utils::Json::array();
        for (const auto& p : coords) {
            if (p.has_value()) {
                utils::Json pt = utils::Json::array();
                pt.push_back(p.value()[0]); pt.push_back(p.value()[1]); pt.push_back(p.value()[2]);
                list.push_back(std::move(pt));
            } else {
                list.push_back(nullptr);
            }
        }
        return list;
    };
    req["prev_u_xyz"] = convert_prev_coords(prev_u);
    req["prev_v_xyz"] = convert_prev_coords(prev_v);
    req["prev_diag_xyz"] = convert_prev_coords(prev_diag);

    utils::Json response = process_json_request(req, sock);

    if (response.contains("error")) {
        throw std::runtime_error("Neural tracer returned error: " + response["error"].get_string());
    }

    auto get_float_or_nan = [](const utils::Json& j) {
        return j.is_null() ? std::numeric_limits<float>::quiet_NaN() : j.get_float();
    };

    auto process_candidates = [&](const utils::Json& batch) {
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
    };

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
