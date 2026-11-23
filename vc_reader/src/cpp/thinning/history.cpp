#include "history.hpp"
#include <array>
#include <cmath>
namespace vcr::thinning {
constexpr int kHistoryLength = 4;
constexpr std::array<int, 8> kNeighborDx = {-1, 0, 1, -1, 1, -1, 0, 1};
constexpr std::array<int, 8> kNeighborDy = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr std::array<std::array<float, kHistoryLength>, kHistoryLength> kHistoryWeights = {{
    {{1.0f, 0.0f, 0.0f, 0.0f}},
    {{1.0f, 0.25f, 0.0f, 0.0f}},
    {{1.0f, 0.625f, 0.25f, 0.0f}},
    {{1.0f, 0.75f, 0.5f, 0.25f}},
}};
constexpr std::array<float, kHistoryLength> kHistoryWeightSums = {
    1.0f,
    1.25f,
    1.875f,
    2.5f,
};

// A history contains at most three 8-connected steps. Cache its eight
// candidate penalties using exactly the same float operations and order.
const detail::Penalties& detail::penalties() {
    static const auto table=[] {
        detail::Penalties result{};
        for(int count=1;count<=4;++count) for(unsigned code=0;code<512;++code) {
            std::array<Point,4> points{};unsigned steps=code;
            for(int i=count-2;i>=0;--i) {
                points[i]={points[i+1].x-kNeighborDx[steps&7],points[i+1].y-kNeighborDy[steps&7]};steps>>=3;
            }
            for(int ni=0;ni<8;++ni) {
                float total=0;
                for(int i=0;i<count;++i) {
                    const float dx=float(kNeighborDx[ni]-points[i].x),dy=float(kNeighborDy[ni]-points[i].y);
                    total+=kHistoryWeights[count-1][i]*std::sqrt(dx*dx+dy*dy);
                }
                result.values[count-1][code][ni]=total/kHistoryWeightSums[count-1];
            }
        }
        return result;
    }();
    return table;
}
}
