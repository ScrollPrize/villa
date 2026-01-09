#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <nlohmann/json.hpp>

#include "vc/tracer/NeuralTracerConnection.h"
#include "vc/core/util/Zarr.hpp"


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

    auto get_float_or_nan = [](const nlohmann::json& j) {
        return j.is_null() ? std::numeric_limits<float>::quiet_NaN() : j.get<float>();
    };

    auto process_candidates = [&](const nlohmann::json& batch) {
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

NeuralTracerConnection::EdtResult NeuralTracerConnection::get_distance_transform(
    const cv::Vec3f& center_xyz,
    const std::string& conditioning_mask_path
) const {
    nlohmann::json req;
    req["request_type"] = "edt";
    req["center_xyz"] = {center_xyz[0], center_xyz[1], center_xyz[2]};
    req["conditioning_mask_path"] = conditioning_mask_path;

    nlohmann::json response = process_json_request(req, sock);

    if (response.contains("error")) {
        throw std::runtime_error("Neural tracer returned error: " + response["error"].get<std::string>());
    }

    if (!response.contains("distance_transform")) {
        throw std::runtime_error("Response missing distance_transform field");
    }

    auto& dt_response = response["distance_transform"];

    EdtResult result;

    // Parse min_corner_xyz
    auto& min_corner = dt_response["min_corner_xyz"];
    result.min_corner_xyz = cv::Vec3f(
        min_corner[0].get<float>(),
        min_corner[1].get<float>(),
        min_corner[2].get<float>()
    );

    // Parse shape
    auto& shape = dt_response["shape"];
    result.shape = {
        shape[0].get<int>(),
        shape[1].get<int>(),
        shape[2].get<int>()
    };

    // Parse scale_factor (default to 1.0 for backward compatibility)
    result.scale_factor = dt_response.value("scale_factor", 1.0f);

    // Parse crop_size (supports both int and array for backward compatibility)
    if (dt_response.contains("crop_size")) {
        auto& cs = dt_response["crop_size"];
        if (cs.is_array()) {
            result.crop_size = {cs[0].get<int>(), cs[1].get<int>(), cs[2].get<int>()};
        } else {
            int val = cs.get<int>();
            result.crop_size = {val, val, val};
        }
    }

    // Read the 3D TIFF file
    std::string zarr_path = dt_response["path"].get<std::string>();
    result.distance_transform = read3DZarr(zarr_path);

    return result;
}

NeuralTracerConnection::EdtResult NeuralTracerConnection::get_sdt_from_points(
    const cv::Vec3f& center_xyz,
    const std::vector<cv::Vec3f>& conditioning_points
) const {
    nlohmann::json req;
    req["request_type"] = "edt";
    req["center_xyz"] = {center_xyz[0], center_xyz[1], center_xyz[2]};

    // Convert conditioning points to JSON array
    nlohmann::json points_json = nlohmann::json::array();
    for (const auto& p : conditioning_points) {
        points_json.push_back({p[0], p[1], p[2]});
    }
    req["conditioning_points_xyz"] = points_json;

    nlohmann::json response = process_json_request(req, sock);

    if (response.contains("error")) {
        throw std::runtime_error("Neural tracer returned error: " + response["error"].get<std::string>());
    }

    if (!response.contains("distance_transform")) {
        throw std::runtime_error("Response missing distance_transform field");
    }

    auto& dt_response = response["distance_transform"];

    EdtResult result;

    // Parse min_corner_xyz
    auto& min_corner = dt_response["min_corner_xyz"];
    result.min_corner_xyz = cv::Vec3f(
        min_corner[0].get<float>(),
        min_corner[1].get<float>(),
        min_corner[2].get<float>()
    );

    // Parse shape
    auto& shape = dt_response["shape"];
    result.shape = {
        shape[0].get<int>(),
        shape[1].get<int>(),
        shape[2].get<int>()
    };

    // Parse scale_factor
    result.scale_factor = dt_response.value("scale_factor", 1.0f);

    // Parse crop_size (supports both int and array for backward compatibility)
    if (dt_response.contains("crop_size")) {
        auto& cs = dt_response["crop_size"];
        if (cs.is_array()) {
            result.crop_size = {cs[0].get<int>(), cs[1].get<int>(), cs[2].get<int>()};
        } else {
            int val = cs.get<int>();
            result.crop_size = {val, val, val};
        }
    }

    // Read the 3D TIFF file
    std::string zarr_path = dt_response["path"].get<std::string>();
    result.distance_transform = read3DZarr(zarr_path);

    return result;
}

NeuralTracerConnection::BatchEdtResult NeuralTracerConnection::get_sdt_from_points_batch(
    const std::vector<cv::Vec3f>& center_xyzs,
    const std::vector<std::vector<cv::Vec3f>>& conditioning_points_list
) const {
    if (center_xyzs.size() != conditioning_points_list.size()) {
        throw std::invalid_argument("center_xyzs and conditioning_points_list must have same size");
    }

    nlohmann::json req;
    req["request_type"] = "edt_batch";

    // Build batch arrays
    nlohmann::json centers_json = nlohmann::json::array();
    nlohmann::json points_batch_json = nlohmann::json::array();

    for (size_t i = 0; i < center_xyzs.size(); ++i) {
        const auto& center = center_xyzs[i];
        centers_json.push_back({center[0], center[1], center[2]});

        nlohmann::json points_json = nlohmann::json::array();
        for (const auto& p : conditioning_points_list[i]) {
            points_json.push_back({p[0], p[1], p[2]});
        }
        points_batch_json.push_back(points_json);
    }

    req["center_xyz_batch"] = centers_json;
    req["conditioning_points_xyz_batch"] = points_batch_json;

    nlohmann::json response = process_json_request(req, sock);

    if (response.contains("error")) {
        throw std::runtime_error("Neural tracer returned error: " + response["error"].get<std::string>());
    }

    if (!response.contains("batch_results")) {
        throw std::runtime_error("Response missing batch_results field");
    }

    // Parse batch response
    BatchEdtResult batch_result;
    auto& results_json = response["batch_results"];
    batch_result.results.resize(results_json.size());
    batch_result.errors.resize(results_json.size());

    for (size_t i = 0; i < results_json.size(); ++i) {
        auto& item = results_json[i];

        if (item.contains("error")) {
            batch_result.errors[i] = item["error"].get<std::string>();
            continue;
        }

        batch_result.valid_indices.push_back(static_cast<int>(i));
        batch_result.errors[i] = "";

        auto& dt_data = item["distance_transform"];

        EdtResult& result = batch_result.results[i];

        // Parse min_corner_xyz
        auto& min_corner = dt_data["min_corner_xyz"];
        result.min_corner_xyz = cv::Vec3f(
            min_corner[0].get<float>(),
            min_corner[1].get<float>(),
            min_corner[2].get<float>()
        );

        // Parse shape
        auto& shape = dt_data["shape"];
        result.shape = {shape[0].get<int>(), shape[1].get<int>(), shape[2].get<int>()};
        result.scale_factor = dt_data.value("scale_factor", 1.0f);
        if (dt_data.contains("crop_size")) {
            auto& cs = dt_data["crop_size"];
            if (cs.is_array()) {
                result.crop_size = {cs[0].get<int>(), cs[1].get<int>(), cs[2].get<int>()};
            } else {
                int val = cs.get<int>();
                result.crop_size = {val, val, val};
            }
        }

        // Read zarr volume
        std::string zarr_path = dt_data["path"].get<std::string>();
        result.distance_transform = read3DZarr(zarr_path);
    }

    return batch_result;
}

NeuralTracerConnection::BatchEdtResult NeuralTracerConnection::get_distance_transform_batch(
    const std::vector<cv::Vec3f>& center_xyzs,
    const std::vector<std::string>& conditioning_mask_paths
) const {
    if (center_xyzs.size() != conditioning_mask_paths.size()) {
        throw std::invalid_argument("center_xyzs and conditioning_mask_paths must have same size");
    }

    nlohmann::json req;
    req["request_type"] = "edt_batch";

    // Build batch arrays
    nlohmann::json centers_json = nlohmann::json::array();
    nlohmann::json masks_json = nlohmann::json::array();

    for (size_t i = 0; i < center_xyzs.size(); ++i) {
        const auto& center = center_xyzs[i];
        centers_json.push_back({center[0], center[1], center[2]});
        masks_json.push_back(conditioning_mask_paths[i]);
    }

    req["center_xyz_batch"] = centers_json;
    req["conditioning_mask_paths"] = masks_json;

    nlohmann::json response = process_json_request(req, sock);

    if (response.contains("error")) {
        throw std::runtime_error("Neural tracer returned error: " + response["error"].get<std::string>());
    }

    if (!response.contains("batch_results")) {
        throw std::runtime_error("Response missing batch_results field");
    }

    // Parse batch response
    BatchEdtResult batch_result;
    auto& results_json = response["batch_results"];
    batch_result.results.resize(results_json.size());
    batch_result.errors.resize(results_json.size());

    for (size_t i = 0; i < results_json.size(); ++i) {
        auto& item = results_json[i];

        if (item.contains("error")) {
            batch_result.errors[i] = item["error"].get<std::string>();
            continue;
        }

        batch_result.valid_indices.push_back(static_cast<int>(i));
        batch_result.errors[i] = "";

        auto& dt_data = item["distance_transform"];

        EdtResult& result = batch_result.results[i];

        // Parse min_corner_xyz
        auto& min_corner = dt_data["min_corner_xyz"];
        result.min_corner_xyz = cv::Vec3f(
            min_corner[0].get<float>(),
            min_corner[1].get<float>(),
            min_corner[2].get<float>()
        );

        // Parse shape
        auto& shape = dt_data["shape"];
        result.shape = {shape[0].get<int>(), shape[1].get<int>(), shape[2].get<int>()};
        result.scale_factor = dt_data.value("scale_factor", 1.0f);
        if (dt_data.contains("crop_size")) {
            auto& cs = dt_data["crop_size"];
            if (cs.is_array()) {
                result.crop_size = {cs[0].get<int>(), cs[1].get<int>(), cs[2].get<int>()};
            } else {
                int val = cs.get<int>();
                result.crop_size = {val, val, val};
            }
        }

        // Read zarr volume
        std::string zarr_path = dt_data["path"].get<std::string>();
        result.distance_transform = read3DZarr(zarr_path);
    }

    return batch_result;
}
