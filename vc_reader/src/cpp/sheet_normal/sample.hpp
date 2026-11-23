// Sheet normals at XYZ points of a volume, reading the context each one needs.
#pragma once
#include "normal.hpp"
#include "../array.hpp"
#include <cstring>
#include <vector>
#ifdef VC_READER_CUDA
#include "../reader_cuda/reader.hpp"
#endif
namespace vcr::sheet_normal {
// shape: the volume's, ZYX. Output: n * 4 floats. Queries without full context give zeros.
template<class T>
void sample(Array& array,const std::int64_t* shape,const double* xyz,std::size_t n,float* out,std::size_t batch=8) {
    const std::size_t voxels=std::size_t(width)*width*width;
    for(std::size_t first=0;first<n;first+=batch) {
        const std::size_t count=std::min(batch,n-first);
        std::vector<T> patches(count*voxels);
        std::vector<float> centers(count*4);
        std::vector<Index> starts,sizes;
        std::vector<std::uint8_t*> buffers;
        for(std::size_t i=0;i<count;++i) {
            std::int64_t start[3];float offset[3];bool valid;
            region(xyz+3*(first+i),shape,start,offset,&valid);
            center(offset,valid,centers.data()+4*i);
            if(valid) {
                starts.emplace_back(start,start+3);
                sizes.push_back({width,width,width});
                buffers.push_back(reinterpret_cast<std::uint8_t*>(patches.data()+i*voxels));
            }
        }
        if(!starts.empty()) array.read_many(starts,sizes,buffers);
        cpu(patches.data(),centers.data(),out+4*first,count);
    }
}
#ifdef VC_READER_CUDA
// The same on the device: context through the reader's brick cache, normals in device
// storage. Reads complete before return.
inline device::Storage sample_cuda(reader_cuda::Reader& reader,const std::int64_t* shape,const double* xyz,std::size_t n,int id,device::Storage output={}) {
    std::vector<reader_cuda::Grid> grids(n);
    std::vector<float> centers(n*4);
    for(std::size_t i=0;i<n;++i) {
        std::int64_t start[3];float offset[3];bool valid;
        region(xyz+3*i,shape,start,offset,&valid);
        center(offset,valid,centers.data()+4*i);
        const double first[3]{double(start[0]),double(start[1]),double(start[2])};
        const auto local=localOrigin(first);
        auto& grid=grids[i];
        for(int a=0;a<3;++a) {
            grid.shape[a]=width;grid.base[a]=local.base[a];grid.origin[a]=local.origin[a];
            grid.axes[4*a]=1;
        }
    }
    auto patches=reader.grids(std::move(grids),1,0,false,"chunks");
    device::Guard guard(id);
    return cuda(patches.storage->ptr,false,centers.data(),n,id,std::move(output));
}
#endif
}
