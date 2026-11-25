#include "common.hpp"
namespace vcr::distance {
namespace {
// min(value, neighbour + 1) where 255 + 1 never wins: a branch-free, vectorizable saturating step.
inline std::uint8_t step(std::uint8_t value, std::uint8_t neighbour) {
    const auto next=std::uint8_t(neighbour+(neighbour!=255));
    return value<next ? value : next;
}
}
void manhattan(const std::uint8_t* input, std::uint8_t* output, const std::vector<std::size_t>& shape) {
    const auto n=detail::validate(input,output,shape);
    if (!n) return;
    for (std::size_t i=0;i<n;++i) output[i]=input[i] ? 255 : 0;
    std::size_t stride=1;
    for (std::size_t a=shape.size();a-->0;) {
        const auto extent=shape[a];
        if (stride==1) {
            // detail::line on groups of rows, interleaved: independent chains instead of one
            // serial dependency per row; each row's arithmetic is unchanged.
            constexpr std::size_t group=16;
            std::size_t i=0;
            for(;i+group*extent<=n;i+=group*extent) {
                std::uint8_t previous[group];
                std::fill_n(previous,group,std::uint8_t(255));
                for(std::size_t q=0;q<extent;++q) for(std::size_t r=0;r<group;++r) {
                    auto& v=output[i+r*extent+q]; previous[r]=v<previous[r]+1 ? v : std::uint8_t(previous[r]+1); v=previous[r];
                }
                std::fill_n(previous,group,std::uint8_t(255));
                for(std::size_t q=extent;q-->0;) for(std::size_t r=0;r<group;++r) {
                    auto& v=output[i+r*extent+q]; previous[r]=v<previous[r]+1 ? v : std::uint8_t(previous[r]+1); v=previous[r];
                }
            }
            for(;i<n;i+=extent) detail::line(output+i,extent,1);
        } else {
            for(std::size_t base=0;base<n;base+=extent*stride) {
                for(std::size_t q=1;q<extent;++q) {
                    auto* row=output+base+q*stride;
                    const auto* previous=row-stride;
                    for(std::size_t i=0;i<stride;++i) row[i]=step(row[i],previous[i]);
                }
                for(std::size_t q=extent-1;q-->0;) {
                    auto* row=output+base+q*stride;
                    const auto* next=row+stride;
                    for(std::size_t i=0;i<stride;++i) row[i]=step(row[i],next[i]);
                }
            }
        }
        stride*=extent;
    }
}
}
