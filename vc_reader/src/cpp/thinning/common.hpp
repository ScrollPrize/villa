#pragma once
#include <vc_reader/thinning.hpp>
#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>
#include <stdexcept>

namespace vcr::thinning::detail {
inline std::size_t validate(const void* input, std::size_t rows, std::size_t cols, std::size_t slices) {
    if (rows > INT_MAX || cols > INT_MAX || (rows && cols > SIZE_MAX / rows) ||
        (rows * cols && slices > SIZE_MAX / (rows * cols)))
        throw std::invalid_argument("thinning dimensions overflow");
    const auto n = rows * cols * slices;
    if (n && !input) throw std::invalid_argument("null thinning input");
    return n;
}
inline void validateAlgorithm(Algorithm a) {
    if (a != Algorithm::ZhangSuen && a != Algorithm::GuoHall && a != Algorithm::ZhangSuenGuoHall)
        throw std::invalid_argument("invalid thinning algorithm");
}
struct View {
    std::uint8_t* data;
    int rows, cols;
    template<class T> T* ptr(int y) const { return reinterpret_cast<T*>(data + std::size_t(y) * cols); }
};
inline double elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
void prepare(const std::uint8_t*, int, int, std::size_t, Scratch&, Stats&);
#ifdef VC_READER_CUDA
void traceCudaHost(const std::uint8_t*,std::size_t,std::size_t,std::uint8_t*,Traces*,Stats*,PruneParams,std::size_t);
#endif
}
