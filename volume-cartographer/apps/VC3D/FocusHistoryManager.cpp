#include "FocusHistoryManager.hpp"
#include <cmath>
#include <algorithm>

void FocusHistoryManager::record(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& surfaceId)
{
    if (_navigating) {
        return;
    }

    Entry entry{position, normal, surfaceId};

    if (_index >= 0 && _index < static_cast<int>(_history.size())) {
        const auto& current = _history[_index];
        const float positionDelta = cv::norm(current.position - entry.position);
        const float normalDelta = cv::norm(current.normal - entry.normal);
        if (positionDelta < 1e-4f && normalDelta < 1e-4f && current.surfaceId == entry.surfaceId) {
            return;
        }
    }

    // Erase forward history if we're not at the end
    if (_index >= 0 && _index + 1 < static_cast<int>(_history.size())) {
        _history.erase(_history.begin() + _index + 1, _history.end());
    }

    _history.push_back(entry);

    if (_history.size() > MAX_HISTORY) {
        _history.pop_front();
        if (_index > 0) {
            --_index;
        }
    }

    _index = static_cast<int>(_history.size()) - 1;
}

std::optional<FocusHistoryManager::Entry> FocusHistoryManager::step(int direction)
{
    if (_history.empty() || direction == 0 || _index < 0) {
        return std::nullopt;
    }

    const int lastIndex = static_cast<int>(_history.size()) - 1;
    int targetIndex = _index + direction;
    targetIndex = std::max(0, std::min(targetIndex, lastIndex));

    if (targetIndex == _index) {
        return std::nullopt;
    }

    _index = targetIndex;
    return _history[_index];
}

void FocusHistoryManager::clear()
{
    _history.clear();
    _index = -1;
    _navigating = false;
}
