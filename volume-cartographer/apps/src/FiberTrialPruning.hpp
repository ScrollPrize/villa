#pragma once
#include <map>
#include <mutex>
#include <limits>
#include <string>

// A model-local replay owns this table. Directions never share pruning state.
class FiberTrialPruning {
    struct State {
        std::map<double, bool> valid;
        double cutoff = std::numeric_limits<double>::infinity();
    };
    std::mutex mutex_;
    std::map<std::string, State> states_;
public:
    bool skip(const std::string& key, double distance) {
        std::lock_guard lock(mutex_);
        return distance > states_[key].cutoff;
    }
    void record(const std::string& key, double distance, bool valid) {
        std::lock_guard lock(mutex_);
        auto& s = states_[key];
        s.valid[distance] = valid;
        if (!valid) return;
        for (double lo : {distance - 100.0, distance}) {
            auto a = s.valid.find(lo), b = s.valid.find(lo + 100.0);
            if (a != s.valid.end() && b != s.valid.end() && a->second && b->second)
                s.cutoff = std::min(s.cutoff, lo + 100.0);
        }
    }
};
