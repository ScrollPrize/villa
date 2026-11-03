#include "normal.hpp"
#include "../cuda_buffer.hpp"
#include <cub/device/device_segmented_radix_sort.cuh>
namespace vcr::sheet_normal {
template<class T> __global__ void filter_z(const T* src,float* dst,Kernels k,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*4*zcount) return;
    int b=i/(4*zcount), c=(i/zcount)%4, q=i%zcount;
    int z=q/(width*width), xy=q%(width*width);
    float sum=0;
    for(int j=0;j<13;++j) sum+=k.taps[c][j]*(float(src[b*width*width*width+(3*z+j)*width*width+xy])*(1.f/255.f));
    dst[i]=sum;
}
__global__ void filter_y(const float* src,float* dst,Kernels k,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*6*ycount) return;
    int b=i/(6*ycount), c=(i/ycount)%6, q=i%ycount;
    int x=q%width, y=(q/width)%sites, z=q/(width*sites);
    int zc=2*(c/3)+(c%3==2), yc=2*(c/3)+(c%3==1);
    float sum=0;
    for(int j=0;j<13;++j) sum+=k.taps[yc][j]*src[(b*4+zc)*zcount+(z*width+3*y+j)*width+x];
    dst[i]=sum;
}
__global__ void filter_x(const float* src,float* dst,Kernels k,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*6*count) return;
    int b=i/(6*count), c=(i/count)%6, q=i%count;
    int xc=2*(c/3)+(c%3==0), x=q%sites, row=q/sites;
    float sum=0;
    for(int j=0;j<13;++j) sum+=k.taps[xc][j]*src[(b*6+c)*ycount+row*width+3*x+j];
    dst[i]=sum;
}
__global__ void magnitudes(const float* g,float* mags,int* offsets,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<=n) offsets[i]=i*count;
    if(i>=n*count) return;
    int b=i/count,q=i%count;
    mags[i]=magnitude(g[(b*6+3)*count+q],g[(b*6+4)*count+q],g[(b*6+5)*count+q]);
}
constexpr int parts=32, threads=128;
__global__ void integrate(const float* g,const float* centers,const float* sorted,
                          const double* normals,double* partial,int pass) {
    int b=blockIdx.y, part=blockIdx.x;
    __shared__ double scratch[7][threads];
    double sums[7]{}, normal[3]{};
    for(int j=0;j<3;++j) normal[j]=normals[4*b+j];
    float floor=sorted[b*count+count/2]; if(floor<1e-8f) floor=1e-8f;
    if(centers[4*b+3] && (!pass || normals[4*b+3]>0))
        for(int i=part*threads+threadIdx.x;i<count;i+=parts*threads)
            contribution(i,g+(b*6+(pass ? 3 : 0))*count,centers+4*b,normal,floor,pass!=0,sums);
    for(int j=0;j<7;++j) scratch[j][threadIdx.x]=sums[j];
    __syncthreads();
    for(int stride=threads/2;stride;stride/=2) {
        if(threadIdx.x<stride) for(int j=0;j<7;++j) scratch[j][threadIdx.x]+=scratch[j][threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0) for(int j=0;j<7;++j) partial[(b*parts+part)*7+j]=scratch[j][0];
}
__global__ void solve(const double* partial,double* normals,float* output,int n) {
    int b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=n) return;
    double sums[7]{},normal[3]{},gap=0;
    for(int p=0;p<parts;++p) for(int j=0;j<7;++j) sums[j]+=partial[(b*parts+p)*7+j];
    bool valid=principal(sums,normal,gap);
    for(int j=0;j<4;++j) { double value=valid ? (j==3 ? gap : normal[j]) : 0.; normals[4*b+j]=value; output[4*b+j]=float(value); }
}
device::Storage cuda(const void* image,bool floating,const float* centers,std::size_t batch,int id,device::Storage out) {
    if(batch>128) throw std::invalid_argument("sheet normal CUDA batch must be <=128; process larger requests in groups");
    if (!out) out=device::allocate(batch*4*sizeof(float),id);
    if(!batch) return out;
    const int n=int(batch); auto stream=device::stream(); Kernels k;
    auto z=device::allocate(n*4*zcount*sizeof(float),id), y=device::allocate(n*6*ycount*sizeof(float),id);
    auto g=device::allocate(n*6*count*sizeof(float),id), mags=device::allocate(n*count*sizeof(float),id);
    auto sorted=device::allocate(n*count*sizeof(float),id), offsets=device::allocate((n+1)*sizeof(int),id);
    auto c=device::allocate(n*4*sizeof(float),id), normals=device::allocate(n*4*sizeof(double),id);
    auto partial=device::allocate(n*parts*7*sizeof(double),id);
    device::check(cudaMemcpyAsync(c->ptr,centers,n*4*sizeof(float),cudaMemcpyHostToDevice,stream));
    device::check(cudaMemsetAsync(normals->ptr,0,n*4*sizeof(double),stream));
    if(floating) filter_z<<<(n*4*zcount+255)/256,256,0,stream>>>(static_cast<const float*>(image),static_cast<float*>(z->ptr),k,n);
    else filter_z<<<(n*4*zcount+255)/256,256,0,stream>>>(static_cast<const std::uint8_t*>(image),static_cast<float*>(z->ptr),k,n);
    filter_y<<<(n*6*ycount+255)/256,256,0,stream>>>(static_cast<float*>(z->ptr),static_cast<float*>(y->ptr),k,n);
    filter_x<<<(n*6*count+255)/256,256,0,stream>>>(static_cast<float*>(y->ptr),static_cast<float*>(g->ptr),k,n);
    magnitudes<<<(n*count+255)/256,256,0,stream>>>(static_cast<float*>(g->ptr),static_cast<float*>(mags->ptr),static_cast<int*>(offsets->ptr),n);
    std::size_t bytes=0;
    device::check(cub::DeviceSegmentedRadixSort::SortKeys(nullptr,bytes,static_cast<float*>(mags->ptr),static_cast<float*>(sorted->ptr),n*count,n,
        static_cast<int*>(offsets->ptr),static_cast<int*>(offsets->ptr)+1,0,32,stream));
    auto temp=device::allocate(bytes,id);
    device::check(cub::DeviceSegmentedRadixSort::SortKeys(temp->ptr,bytes,static_cast<float*>(mags->ptr),static_cast<float*>(sorted->ptr),n*count,n,
        static_cast<int*>(offsets->ptr),static_cast<int*>(offsets->ptr)+1,0,32,stream));
    for(int pass=0;pass<4;++pass) {
        integrate<<<dim3(parts,n),threads,0,stream>>>(static_cast<float*>(g->ptr),static_cast<float*>(c->ptr),static_cast<float*>(sorted->ptr),
            static_cast<double*>(normals->ptr),static_cast<double*>(partial->ptr),pass);
        solve<<<(n+31)/32,32,0,stream>>>(static_cast<double*>(partial->ptr),static_cast<double*>(normals->ptr),static_cast<float*>(out->ptr),n);
    }
    device::check(cudaGetLastError()); device::complete();
    return out;
}
}
