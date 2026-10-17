#include "fourier.cuh"
namespace vcr::augment::cuda {
namespace vd=volume_detail;
__global__ void make_complex(const double* src,vd::Complex* dest,std::size_t n){EACH_INDEX(n)dest[i]={src[i],0};}
__global__ void real_part(const vd::Complex* src,double* dest,std::size_t n){EACH_INDEX(n)dest[i]=src[i].re;}
__global__ void fft_lines(vd::Complex* data,vd::Grid3 g,int axis,vd::Complex* scratch,std::size_t m,std::size_t start,std::size_t count,bool inverse){
    EACH_INDEX(count){auto line=start+i,stride=g.stride[axis],len=g.n[axis],base=(line/stride)*len*stride+line%stride;vd::fft_line(data+base,len,stride,scratch+i*2*m,m,inverse);}
}
void fft_volume(Buffer<vd::Complex>& data,const vd::Grid3& g,bool inverse){
    for(int axis=0;axis<3;++axis){auto len=g.n[axis],count=g.size/len,size=vd::fft_scratch(len);auto batch=std::min<std::size_t>(count,std::max<std::size_t>(1,(16*1024*1024)/(size*sizeof(vd::Complex))));Buffer<vd::Complex> scratch(batch*size);
        for(std::size_t first=0;first<count;first+=batch){auto n=std::min(batch,count-first);fft_lines<<<blocks(n),256,0,vcr::device::stream()>>>(data.ptr,g,axis,scratch.ptr,size/2,first,n,inverse);check(cudaGetLastError());}}
}
void to_complex(View<double> input,Buffer<vd::Complex>& output){make_complex<<<blocks(input.size),256,0,vcr::device::stream()>>>(input.ptr,output.ptr,input.size);check(cudaGetLastError());}
void from_complex(const Buffer<vd::Complex>& input,View<double> output){real_part<<<blocks(output.size),256,0,vcr::device::stream()>>>(input.ptr,output.ptr,output.size);check(cudaGetLastError());}
}
