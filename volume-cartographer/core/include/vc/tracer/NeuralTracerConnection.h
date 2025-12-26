#pragma once

#include <string>
#include <optional>
#include <opencv2/core.hpp>


class NeuralTracerConnection
{
public:

    struct NextUvs
    {
        std::vector<cv::Vec3f> next_u_xyzs;
        std::vector<cv::Vec3f> next_v_xyzs;
    };

    explicit NeuralTracerConnection(std::string const & socket_path);
    ~NeuralTracerConnection();

    NeuralTracerConnection(NeuralTracerConnection const &) = delete;
    NeuralTracerConnection &operator =(NeuralTracerConnection const &) = delete;

    std::vector<NextUvs> get_next_points(
        std::vector<cv::Vec3f> const &center,
        std::vector<std::optional<cv::Vec3f>> const &prev_u,
        std::vector<std::optional<cv::Vec3f>> const &prev_v,
        std::vector<std::optional<cv::Vec3f>> const &prev_diag
    ) const;

private:
    int sock = -1;
};
