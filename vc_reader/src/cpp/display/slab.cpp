#include "slab_pixel.hpp"
#include "../thread_pool.hpp"
#include <thread>
#include <stdexcept>
namespace vcr::display {
std::array<float,256> opacityLut(float low,float high,float opacity,float gamma,uint8_t cutoff,float segmentLength) {
    std::array<float,256> out{};
    for(int i=0;i<256;++i)out[i]=detail::opacityValue(i,low,high,opacity,gamma,cutoff,segmentLength);
    return out;
}
void compositeSlab(const Slab& slab,uint8_t* rgb,uint8_t* coverage) {
    if(slab.width<=0 || slab.height<=0)return;
    if(!rgb || !coverage || !slab.layers || !slab.colors || !slab.opacity || slab.count<=0 ||
        (slab.projection.perspective ? !slab.projection.layerNum : !slab.projection.layerOffsets))
        throw std::invalid_argument("slab requires layers, palettes and output buffers");
    static ThreadPool pool(std::max(1u,std::thread::hardware_concurrency()));
    pool.parallel_for(std::size_t(slab.height),[&](std::size_t y) {
        for(int x=0;x<slab.width;++x) {
            const auto i=y*slab.width+x;
            detail::slabPixel(slab,x,int(y),rgb+3*i,coverage+i);
        }
    });
}
}
