#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

namespace vc::objects::detail {
// Shared broad phase for resident database regions and in-memory surface tiles.
class BoundsIndex {
public:
    using Box = std::array<double, 6>;

    void build(std::vector<Box> boxes)
    {
        boxes_ = std::move(boxes);
        nodes_.clear();
        order_.resize(boxes_.size());
        std::iota(order_.begin(), order_.end(), size_t{0});
        nodes_.reserve(boxes_.size() * 2);
        if (!boxes_.empty()) split(0, boxes_.size());
    }

    template<class Visitor>
    void query(const Box& box, Visitor visit) const
    {
        if (nodes_.empty()) return;
        auto walk = [&](auto&& self, size_t index) -> void {
            const auto& node = nodes_[index];
            if (!overlaps(node.box, box)) return;
            if (node.left) {
                self(self, node.left);
                self(self, node.right);
            } else {
                for (size_t j = node.begin; j < node.end; ++j) {
                    if (overlaps(boxes_[order_[j]], box)) visit(order_[j]);
                }
            }
        };
        walk(walk, 0);
    }

private:
    struct Node {
        Box box;
        size_t begin, end, left = 0, right = 0;
    };
    std::vector<Box> boxes_;
    std::vector<size_t> order_;
    std::vector<Node> nodes_;

    static bool overlaps(const Box& a, const Box& b)
    {
        for (size_t axis = 0; axis < 3; ++axis) {
            if (a[axis] > b[axis + 3] || a[axis + 3] < b[axis]) return false;
        }
        return true;
    }

    size_t split(size_t begin, size_t end)
    {
        auto box = boxes_[order_[begin]];
        for (size_t i = begin + 1; i < end; ++i) {
            for (size_t axis = 0; axis < 3; ++axis) {
                box[axis] = std::min(box[axis], boxes_[order_[i]][axis]);
                box[axis + 3] = std::max(box[axis + 3], boxes_[order_[i]][axis + 3]);
            }
        }
        const size_t index = nodes_.size();
        nodes_.push_back({box, begin, end});
        if (end - begin > 8) {
            size_t axis = 0;
            for (size_t a = 1; a < 3; ++a) {
                if (box[a + 3] / 2 - box[a] / 2 > box[axis + 3] / 2 - box[axis] / 2) axis = a;
            }
            const size_t middle = begin + (end - begin) / 2;
            std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
                [&](size_t a, size_t b) {
                    const auto x = std::midpoint(boxes_[a][axis], boxes_[a][axis + 3]);
                    const auto y = std::midpoint(boxes_[b][axis], boxes_[b][axis + 3]);
                    return x < y || (x == y && a < b);
                });
            const size_t left = split(begin, middle), right = split(middle, end);
            nodes_[index].left = left;
            nodes_[index].right = right;
        }
        return index;
    }
};
} // namespace vc::objects::detail
