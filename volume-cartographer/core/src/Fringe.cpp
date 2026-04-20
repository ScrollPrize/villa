#include "vc/core/util/Fringe.hpp"

#include <algorithm>
#include <limits>

namespace vc::surface_helpers {

Fringe::Fringe() = default;

void Fringe::init(cv::Size grid_size, const Config& config) {
    _size = grid_size;
    _config = config;
    _attempts = cv::Mat_<uint16_t>(grid_size, static_cast<uint16_t>(0));
    _fringe.clear();
    _at_right_border = false;
    _at_top_border = false;
    _at_bottom_border = false;
}

void Fringe::setContext(const GridContext& ctx) {
    _ctx = ctx;
}

// ---- Core set operations ----

void Fringe::insert(const cv::Vec2i& p) {
    _fringe.insert(p);
}

void Fringe::clear() {
    _fringe.clear();
}

bool Fringe::empty() const {
    return _fringe.empty();
}

size_t Fringe::size() const {
    return _fringe.size();
}

// ---- Attempt tracking ----

void Fringe::incrementAttempts(const cv::Vec2i& p) {
    if (_attempts(p) < std::numeric_limits<uint16_t>::max())
        _attempts(p) = static_cast<uint16_t>(_attempts(p) + 1);
}

void Fringe::resetAttempts(const cv::Vec2i& p) {
    _attempts(p) = 0;
}

void Fringe::resetAllAttempts() {
    _attempts.setTo(0);
}

bool Fringe::shouldPrune(const cv::Vec2i& p) const {
    return _config.max_attempts > 0 && _attempts(p) >= _config.max_attempts;
}

uint16_t Fringe::getAttempts(const cv::Vec2i& p) const {
    return _attempts(p);
}

// ---- Border tracking ----

void Fringe::checkBorderContact(const cv::Vec2i& pn, cv::Size grid_size) {
    if (pn[1] >= grid_size.width)
        _at_right_border = true;
    if (pn[0] < 0)
        _at_top_border = true;
    if (pn[0] >= grid_size.height)
        _at_bottom_border = true;
}

void Fringe::clearBorderFlags() {
    _at_right_border = false;
    _at_top_border = false;
    _at_bottom_border = false;
}

// ---- Grid resize ----

void Fringe::resize(cv::Size new_size, cv::Rect copy_roi) {
    cv::Mat_<uint16_t> new_attempts(new_size, static_cast<uint16_t>(0));
    if (copy_roi.width > 0 && copy_roi.height > 0) {
        cv::Rect src_roi(0, 0, std::min(copy_roi.width, _attempts.cols),
                         std::min(copy_roi.height, _attempts.rows));
        cv::Rect dst_roi(copy_roi.x, copy_roi.y, src_roi.width, src_roi.height);
        _attempts(src_roi).copyTo(new_attempts(dst_roi));
    }
    _attempts = new_attempts;
    _size = new_size;
}

// ---- Rebuild operations ----

bool Fringe::inBounds(const cv::Vec2i& p) const {
    return p[0] >= 0 && p[0] < _size.height && p[1] >= 0 && p[1] < _size.width;
}

void Fringe::rebuildBoundary() {
    _fringe.clear();
    static const std::vector<cv::Vec2i> neighs4 = {{1,0},{0,1},{-1,0},{0,-1}};
    static const std::vector<cv::Vec2i> neighs8 = {{1,0},{0,1},{-1,0},{0,-1},
                                                   {1,1},{1,-1},{-1,1},{-1,-1}};
    const std::vector<cv::Vec2i>& neighs =
        (_config.neighbor_connectivity == 8) ? neighs8 : neighs4;

    for (int y = _ctx.used_area->y; y < _ctx.used_area->br().y; ++y) {
        for (int x = _ctx.used_area->x; x < _ctx.used_area->br().x; ++x) {
            if (((*_ctx.state)(y, x) & detail::kStateFringeLocValid) == 0)
                continue;
            cv::Vec2i p = {y, x};
            bool is_boundary = false;
            for (const auto& n : neighs) {
                cv::Vec2i pn = p + n;
                if (!inBounds(pn)) {
                    checkBorderContact(pn, _size);
                    continue;
                }
                if (((*_ctx.state)(pn) & detail::kStateFringeLocValid) != 0)
                    continue;
                if (((*_ctx.state)(pn) & detail::kStateFringeProcessing) != 0)
                    continue;
                if (_ctx.is_savable && _ctx.is_savable(pn))
                    is_boundary = true;
            }
            if (is_boundary)
                _fringe.insert(p);
        }
    }
}

void Fringe::rebuildIncremental(int padding) {
    cv::Rect rect = *_ctx.used_area;
    if (_ctx.active_bounds)
        rect = *_ctx.active_bounds & *_ctx.used_area;
    rebuildIncrementalRect(rect, padding);
}

void Fringe::rebuildIncrementalRect(const cv::Rect& rect, int padding) {
    _fringe.clear();
    for (int j = rect.y - padding; j <= rect.br().y + padding; ++j) {
        for (int i = rect.x - padding; i <= rect.br().x + padding; ++i) {
            cv::Vec2i p = {j, i};
            if (!inBounds(p))
                continue;
            if ((*_ctx.state)(p) & detail::kStateFringeLocValid)
                _fringe.insert(p);
        }
    }
}

}  // namespace vc::surface_helpers
