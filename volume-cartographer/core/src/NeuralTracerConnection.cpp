#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <nlohmann/json.hpp>

#include "vc/tracer/NeuralTracerConnection.h"


namespace
{
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

        std::cout << "Received from neural tracer: " << response_str << std::endl;

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

NeuralTracerConnection::NextUvs NeuralTracerConnection::get_next_points(
    cv::Vec3f const &center,
    std::optional<cv::Vec3f> const &prev_u,
    std::optional<cv::Vec3f> const &prev_v,
    std::optional<cv::Vec3f> const &prev_diag
) const {
    nlohmann::json req;

    req["center_xyz"] = {center[0], center[1], center[2]};

    if (prev_u.has_value()) {
        req["prev_u_xyz"] = {prev_u.value()[0], prev_u.value()[1], prev_u.value()[2]};
    }
    if (prev_v.has_value()) {
        req["prev_v_xyz"] = {prev_v.value()[0], prev_v.value()[1], prev_v.value()[2]};
    }
    if (prev_diag.has_value()) {
        req["prev_diag_xyz"] = {prev_diag.value()[0], prev_diag.value()[1], prev_diag.value()[2]};
    }

    nlohmann::json response = process_json_request(req, sock);

    NextUvs result;

    auto get_float_or_nan = [](const nlohmann::json& j) {
        return j.is_null() ? std::numeric_limits<float>::quiet_NaN() : j.get<float>();
    };

    if (response.contains("u_candidates")) {
        for (auto const& u_candidate : response["u_candidates"]) {
            if (u_candidate.is_array() && u_candidate.size() == 3) {
                result.next_u_xyzs.emplace_back(
                    get_float_or_nan(u_candidate[0]),
                    get_float_or_nan(u_candidate[1]),
                    get_float_or_nan(u_candidate[2])
                );
            }
        }
    }
    if (response.contains("v_candidates")) {
        for (auto const& v_candidate : response["v_candidates"]) {
            if (v_candidate.is_array() && v_candidate.size() == 3) {
                result.next_v_xyzs.emplace_back(
                    get_float_or_nan(v_candidate[0]),
                    get_float_or_nan(v_candidate[1]),
                    get_float_or_nan(v_candidate[2])
                );
            }
        }
    }

    return result;
}
