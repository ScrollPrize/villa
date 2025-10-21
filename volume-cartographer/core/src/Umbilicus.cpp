#include "vc/core/util/Umbilicus.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <numbers>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {

std::string TrimCopy(const std::string& value)
{
    auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) { return std::isspace(ch) != 0; });
    auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
}

cv::Vec2f SeamDirectionVector(vc::core::util::Umbilicus::SeamDirection direction)
{
    using SeamDirection = vc::core::util::Umbilicus::SeamDirection;
    switch (direction) {
        case SeamDirection::PositiveX: return {1.0f, 0.0f};
        case SeamDirection::NegativeX: return {-1.0f, 0.0f};
        case SeamDirection::PositiveY: return {0.0f, 1.0f};
        case SeamDirection::NegativeY: return {0.0f, -1.0f};
    }
    throw std::logic_error("Unhandled seam direction");
}

constexpr double DegreesPerRadian = 180.0 / std::numbers::pi_v<double>;

} // namespace

namespace vc::core::util {

Umbilicus Umbilicus::FromFile(const std::filesystem::path& path, const cv::Vec3i& volume_shape)
{
    return Umbilicus(LoadFile(path), volume_shape);
}

Umbilicus Umbilicus::FromPoints(std::vector<cv::Vec3f> control_points, const cv::Vec3i& volume_shape)
{
    return Umbilicus(std::move(control_points), volume_shape);
}

const cv::Vec3i& Umbilicus::volume_shape() const noexcept
{
    return volume_shape_;
}

const std::vector<cv::Vec3f>& Umbilicus::centers() const noexcept
{
    return dense_centers_;
}

const cv::Vec3f& Umbilicus::center_at(int z_index) const
{
    if (z_index < 0 || z_index >= static_cast<int>(dense_centers_.size())) {
        throw std::out_of_range("z_index outside interpolated range");
    }
    return dense_centers_[z_index];
}

cv::Vec3f Umbilicus::vector_to_umbilicus(const cv::Vec3f& point) const
{
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }
    const int index = clamp_z_index(point[2]);
    return dense_centers_[index] - point;
}

double Umbilicus::distance_to_umbilicus(const cv::Vec3f& point) const
{
    return cv::norm(vector_to_umbilicus(point));
}

void Umbilicus::set_seam(SeamDirection direction)
{
    set_seam_direction_xy(SeamDirectionVector(direction), direction);
}

void Umbilicus::set_seam_from_point(const cv::Vec3f& point)
{
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }

    const int index = clamp_z_index(point[2]);
    const auto& center = dense_centers_[index];
    cv::Vec2f direction_xy{point[0] - center[0], point[1] - center[1]};

    if (cv::norm(direction_xy) == 0.0f) {
        throw std::invalid_argument("Cannot derive seam direction from coincident point");
    }

    set_seam_direction_xy(direction_xy, std::nullopt);
}

bool Umbilicus::has_seam() const noexcept
{
    return seam_direction_xy_.has_value();
}

Umbilicus::SeamDirection Umbilicus::seam_direction() const
{
    if (!seam_direction_hint_) {
        throw std::logic_error("Umbilicus seam direction requested but no cardinal seam is set");
    }
    return *seam_direction_hint_;
}

std::pair<cv::Vec3f, cv::Vec3f> Umbilicus::seam_segment(int z_index) const
{
    if (!seam_direction_xy_) {
        throw std::logic_error("Umbilicus seam requested before being set");
    }
    if (z_index < 0 || z_index >= static_cast<int>(dense_centers_.size())) {
        throw std::out_of_range("z_index outside interpolated range");
    }
    return {dense_centers_[z_index], seam_endpoints_[z_index]};
}

const std::vector<cv::Vec3f>& Umbilicus::seam_endpoints() const
{
    if (!seam_direction_xy_) {
        throw std::logic_error("Umbilicus seam requested before being set");
    }
    return seam_endpoints_;
}

double Umbilicus::theta(const cv::Vec3f& point, int wrap_count) const
{
    if (!seam_direction_xy_) {
        throw std::logic_error("Umbilicus seam direction needed to compute theta");
    }
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }

    const int index = clamp_z_index(point[2]);
    const auto& center = dense_centers_[index];

    const cv::Vec2f& base = *seam_direction_xy_;
    cv::Vec2f ray{point[0] - center[0], point[1] - center[1]};

    if (cv::norm(ray) == 0.0f) {
        return static_cast<double>(wrap_count) * 360.0;
    }

    const double det = static_cast<double>(base[0]) * ray[1] - static_cast<double>(base[1]) * ray[0];
    const double dot = static_cast<double>(base[0]) * ray[0] + static_cast<double>(base[1]) * ray[1];
    double angle = std::atan2(det, dot) * DegreesPerRadian;
    if (angle < 0.0) {
        angle += 360.0;
    }
    angle += static_cast<double>(wrap_count) * 360.0;
    return angle;
}

Umbilicus::Umbilicus(std::vector<cv::Vec3f> control_points, const cv::Vec3i& volume_shape)
    : volume_shape_(volume_shape), control_points_(std::move(control_points))
{
    if (volume_shape_[0] <= 0 || volume_shape_[1] <= 0 || volume_shape_[2] <= 0) {
        throw std::invalid_argument("Volume shape components must be positive");
    }
    if (control_points_.empty()) {
        throw std::invalid_argument("Umbilicus requires at least one control point");
    }

    std::sort(control_points_.begin(), control_points_.end(), [](const auto& a, const auto& b) {
        return a[2] < b[2];
    });

    interpolate_centers();
}

std::vector<cv::Vec3f> Umbilicus::LoadFile(const std::filesystem::path& path)
{
    const std::string extension = path.extension().string();
    std::string lowered_ext;
    lowered_ext.resize(extension.size());
    std::transform(extension.begin(), extension.end(), lowered_ext.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (lowered_ext == ".json") {
        return LoadJsonFile(path);
    }

    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open umbilicus text file: " + path.string());
    }
    return LoadTextFile(stream);
}

std::vector<cv::Vec3f> Umbilicus::LoadTextFile(std::istream& stream)
{
    std::vector<cv::Vec3f> points;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(stream, line)) {
        ++line_number;
        auto trimmed = TrimCopy(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        std::array<double, 3> values{};
        std::size_t value_index = 0;

        std::stringstream ss(trimmed);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = TrimCopy(token);
            if (token.empty()) {
                continue;
            }
            if (value_index >= values.size()) {
                throw std::runtime_error("Too many columns in umbilicus text line " + std::to_string(line_number));
            }
            try {
                values[value_index] = std::stod(token);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid numeric value in umbilicus text line " + std::to_string(line_number));
            }
            ++value_index;
        }

        if (value_index != values.size()) {
            throw std::runtime_error("Not enough columns in umbilicus text line " + std::to_string(line_number));
        }

        cv::Vec3f point{
            static_cast<float>(values[2]),
            static_cast<float>(values[1]),
            static_cast<float>(values[0])
        };
        points.push_back(point);
    }

    if (points.empty()) {
        throw std::runtime_error("Umbilicus text file contained no points");
    }

    return points;
}

std::vector<cv::Vec3f> Umbilicus::LoadJsonFile(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open umbilicus json file: " + path.string());
    }

    nlohmann::json document;
    stream >> document;

    const nlohmann::json* array = nullptr;
    if (document.is_array()) {
        array = &document;
    } else if (document.contains("points")) {
        const auto& candidate = document.at("points");
        if (!candidate.is_array()) {
            throw std::runtime_error("'points' member in umbilicus json must be an array");
        }
        array = &candidate;
    } else {
        throw std::runtime_error("Umbilicus json root must be an array or contain a 'points' array");
    }

    std::vector<cv::Vec3f> points;
    points.reserve(array->size());

    for (std::size_t idx = 0; idx < array->size(); ++idx) {
        const auto& entry = (*array)[idx];
        double z = 0.0;
        double y = 0.0;
        double x = 0.0;

        if (entry.is_array()) {
            if (entry.size() < 3) {
                throw std::runtime_error("Umbilicus json entry at index " + std::to_string(idx) + " expected three values");
            }
            z = entry[0].get<double>();
            y = entry[1].get<double>();
            x = entry[2].get<double>();
        } else if (entry.is_object()) {
            if (!entry.contains("z") || !entry.contains("y") || !entry.contains("x")) {
                throw std::runtime_error("Umbilicus json entry at index " + std::to_string(idx) + " missing z/y/x keys");
            }
            z = entry.at("z").get<double>();
            y = entry.at("y").get<double>();
            x = entry.at("x").get<double>();
        } else {
            throw std::runtime_error("Umbilicus json entry at index " + std::to_string(idx) + " has unsupported type");
        }

        points.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }

    if (points.empty()) {
        throw std::runtime_error("Umbilicus json file contained no points");
    }

    return points;
}

void Umbilicus::interpolate_centers()
{
    dense_centers_.clear();
    dense_centers_.resize(volume_shape_[0]);
    for (int z = 0; z < volume_shape_[0]; ++z) {
        dense_centers_[z] = interpolate_center(static_cast<double>(z));
    }
}

cv::Vec3f Umbilicus::interpolate_center(double z) const
{
    if (control_points_.empty()) {
        throw std::logic_error("Umbilicus interpolation requested without control points");
    }

    if (control_points_.size() == 1) {
        cv::Vec3f result = control_points_.front();
        result[2] = static_cast<float>(z);
        return result;
    }

    const double min_z = control_points_.front()[2];
    const double max_z = control_points_.back()[2];

    if (z <= min_z) {
        cv::Vec3f result = control_points_.front();
        result[2] = static_cast<float>(z);
        return result;
    }
    if (z >= max_z) {
        cv::Vec3f result = control_points_.back();
        result[2] = static_cast<float>(z);
        return result;
    }

    const float target = static_cast<float>(z);
    auto upper = std::lower_bound(control_points_.begin(), control_points_.end(), target,
                                  [](const cv::Vec3f& lhs, float value) { return lhs[2] < value; });
    if (upper == control_points_.begin()) {
        cv::Vec3f result = *upper;
        result[2] = static_cast<float>(z);
        return result;
    }

    if (upper != control_points_.end() && std::abs((*upper)[2] - target) < 1e-5f) {
        cv::Vec3f result = *upper;
        result[2] = static_cast<float>(z);
        return result;
    }

    const auto& right = (upper == control_points_.end()) ? control_points_.back() : *upper;
    const auto& left = *(upper - 1);

    const double z0 = left[2];
    const double z1 = right[2];

    if (std::abs(z1 - z0) < 1e-5) {
        cv::Vec3f result = left;
        result[2] = static_cast<float>(z);
        return result;
    }

    const double t = (z - z0) / (z1 - z0);
    const double x = left[0] + t * (right[0] - left[0]);
    const double y = left[1] + t * (right[1] - left[1]);
    return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}

int Umbilicus::clamp_z_index(double z) const
{
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }
    int index = static_cast<int>(std::lround(z));
    index = std::clamp(index, 0, static_cast<int>(dense_centers_.size() - 1));
    return index;
}

void Umbilicus::set_seam_direction_xy(const cv::Vec2f& direction,
                                      std::optional<SeamDirection> hint)
{
    if (dense_centers_.empty()) {
        throw std::logic_error("Umbilicus has no interpolated centers");
    }

    const float length = cv::norm(direction);
    if (length == 0.0f) {
        throw std::invalid_argument("Seam direction must be non-zero");
    }

    seam_direction_xy_ = direction * (1.0f / length);
    seam_direction_hint_ = hint;
    seam_endpoints_.resize(dense_centers_.size());
    compute_seam_endpoints();
}

void Umbilicus::compute_seam_endpoints()
{
    if (!seam_direction_xy_) {
        throw std::logic_error("Seam direction must be set before computing endpoints");
    }

    if (dense_centers_.empty()) {
        seam_endpoints_.clear();
        return;
    }

    const float min_x = 0.0f;
    const float min_y = 0.0f;
    const float max_x = static_cast<float>(volume_shape_[2] - 1);
    const float max_y = static_cast<float>(volume_shape_[1] - 1);

    const cv::Vec2f& dir = *seam_direction_xy_;
    const float dx = dir[0];
    const float dy = dir[1];
    constexpr float eps = 1e-6f;

    for (std::size_t idx = 0; idx < dense_centers_.size(); ++idx) {
        const cv::Vec3f& center = dense_centers_[idx];
        float best_t = std::numeric_limits<float>::infinity();

        auto consider_candidate = [&](float t) {
            if (!std::isfinite(t) || t <= 0.0f) {
                return;
            }
            const float x = center[0] + dx * t;
            const float y = center[1] + dy * t;
            if (x < min_x - 1e-3f || x > max_x + 1e-3f) {
                return;
            }
            if (y < min_y - 1e-3f || y > max_y + 1e-3f) {
                return;
            }
            best_t = std::min(best_t, t);
        };

        if (std::abs(dx) > eps) {
            const float t = (dx > 0.0f)
                                ? (max_x - center[0]) / dx
                                : (min_x - center[0]) / dx;
            consider_candidate(t);
        }

        if (std::abs(dy) > eps) {
            const float t = (dy > 0.0f)
                                ? (max_y - center[1]) / dy
                                : (min_y - center[1]) / dy;
            consider_candidate(t);
        }

        if (!std::isfinite(best_t)) {
            best_t = 0.0f;
        }

        cv::Vec3f endpoint = center;
        endpoint[0] = std::clamp(center[0] + dx * best_t, min_x, max_x);
        endpoint[1] = std::clamp(center[1] + dy * best_t, min_y, max_y);
        seam_endpoints_[idx] = endpoint;
    }
}

} // namespace vc::core::util
