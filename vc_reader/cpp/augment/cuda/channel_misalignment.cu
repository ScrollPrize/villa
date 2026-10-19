#include "../channel_misalignment.hpp"
#include "geometry.cuh"
namespace vcr::augment::cuda {
namespace vd=volume_detail;
void apply_cuda(const ChannelMisalignment& t,Sample& s,const Parameters& p,Session& session){
    for(const auto& c:p.channels){auto& a=session.arrays.at(c.key);auto shape=a.shape;shape[0]=1;auto n=product(a.spatial());void* pointer=a.buffer->ptr+c.channel*n*a.bytes;DeviceArray one(shape,a.kind,a.bytes,pointer);auto q=vd::channel_geometry(c,a.spatial(),t.affine.border);auto out=resample_cuda(one,q,false,s.at(c.key).target.fill,false,nullptr);check(copy_async(pointer,out.buffer->ptr,n*a.bytes,cudaMemcpyDeviceToDevice,vcr::device::stream()));}session.restore_valid(s);
}
}
