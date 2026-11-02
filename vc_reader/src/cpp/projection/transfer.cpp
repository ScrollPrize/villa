#include "transfer.hpp"
#include <stdexcept>
#include "../thread_pool.hpp"
namespace vcr::projection {
void transfer(const RasterRequest& r,RasterResult& out,int device){
    if(r.target_height<2||r.target_width<2||r.source_height<2||r.source_width<2||r.output_height<1||r.output_width<1||r.label_height<1||r.label_width<1||r.row_start<0||r.col_start<0||r.row_end>r.output_height||r.col_end>r.output_width||r.row_end<=r.row_start||r.col_end<=r.col_start||!std::isfinite(r.max_distance)||r.max_distance<=0)throw std::invalid_argument("invalid surface transfer bounds or distance");
#ifdef VC_READER_CUDA
    if(device>=0){transferCuda(r,out,device);return;}
#else
    if(device>=0)throw std::runtime_error("vc_reader built without CUDA transfer");
#endif
    out={};auto n=(r.row_end-r.row_start)*(r.col_end-r.col_start);
    if(n<4096){for(std::int64_t i=0;i<n;++i)transferPixel(r,&out,i);return;}
    static ThreadPool pool(std::min(8u,std::max(1u,std::thread::hardware_concurrency())));
    std::vector<RasterResult> counts((n+4095)/4096);
    pool.parallel_for(counts.size(),[&](std::size_t block){
        for(auto i=std::int64_t(block)*4096;i<std::min(n,std::int64_t(block+1)*4096);++i)transferPixel(r,&counts[block],i);
    });
    for(const auto& c:counts){
        out.target_surface_valid+=c.target_surface_valid;
        out.measured_pixels+=c.measured_pixels;
        out.seam_filled_pixels+=c.seam_filled_pixels;
        out.seam_blocked_pixels+=c.seam_blocked_pixels;
        out.inherited_filled_pixels+=c.inherited_filled_pixels;
    }
}
}
