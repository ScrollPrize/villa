#include <vc_reader/pathfinding.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#ifdef VC_READER_CUDA
#include <cuda_runtime.h>
#endif
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
using Path = std::vector<std::uint32_t>;
using Search = Path(*)(const float*,std::size_t,std::size_t,std::uint32_t,std::uint32_t);
void verify(Search search) {
    std::mt19937 rng(77);
    constexpr float inf = std::numeric_limits<float>::infinity();
    // Independent Bellman-Ford, dense graph of all pairs, integer-valued costs.
    for (int trial = 0; trial < 100; ++trial) {
        const int h = 1 + rng()%8, w = 1 + rng()%8, n = h*w;
        std::vector<float> costs(n);
        for (auto& c : costs) c = rng()%5 == 0 ? inf : float(rng()%12);
        const unsigned source = rng()%n, target = rng()%n;
        std::vector<double> distances(n, inf);
        if (std::isfinite(costs[source])) distances[source] = 0;
        for (int round = 0; round < n; ++round)
            for (int from = 0; from < n; ++from) for (int to = 0; to < n; ++to)
                if (std::abs(from/w-to/w)<=1 && std::abs(from%w-to%w)<=1)
                    distances[to] = std::min(distances[to], distances[from]+costs[to]);
        const auto path = search(costs.data(),h,w,source,target);
        CHECK(path == search(costs.data(),h,w,source,target));
        if (!std::isfinite(distances[target])) { CHECK(path.empty()); continue; }
        CHECK(!path.empty() && path.front()==source && path.back()==target);
        double cost=0;
        for (std::size_t i=1;i<path.size();++i) {
            CHECK(std::abs(int(path[i]/w)-int(path[i-1]/w))<=1);
            CHECK(std::abs(int(path[i]%w)-int(path[i-1]%w))<=1);
            cost+=costs[path[i]];
        }
        CHECK(cost==distances[target]);
    }
    float costs[]{1,1,1,1};
    CHECK(search(costs,2,2,0,3)==Path({0,3}));
    CHECK(search(costs,2,2,0,0)==Path({0}));
    for (float invalid : {-1.f, -inf, std::numeric_limits<float>::quiet_NaN()}) {
        costs[3]=invalid;
        bool threw=false; try { search(costs,2,2,0,0); } catch(const std::invalid_argument&) { threw=true; }
        CHECK(threw);
    }
    costs[3]=1;
    bool threw=false; try { search(costs,2,2,0,4); } catch(const std::out_of_range&) { threw=true; }
    CHECK(threw);
}
int main() {
    verify(vcr::pathfinding::shortestPath);
#ifdef VC_READER_CUDA
    int count=0;
    if(cudaGetDeviceCount(&count)==cudaSuccess && count) verify(vcr::pathfinding::cuda::shortestPath);
    else std::cout << "CUDA unavailable; CPU checks passed\n";
#endif
}
