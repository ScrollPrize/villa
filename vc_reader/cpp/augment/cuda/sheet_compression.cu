#include "geometry.cuh"
#include "gaussian.cuh"
namespace vcr::augment::cuda {
template<class T> __device__ void compression_prefix(KernelIndex vcr_launch, T* values,Layout g,int axis,double strength,double threshold,const Statistics* stats) {
    const double span=stats->high-stats->low; const auto step=g.stride[axis],extent=g.shape[axis],lines=g.size/extent;
    EACH_INDEX(lines) {
        const auto base=(i/step)*extent*step+i%step; double sum=0;
        for(std::size_t x=0;x<extent;++x) {
            const auto j=base+x*step; const double norm=(values[j]-stats->low)/span;
            sum += span<=1e-8 ? 0 : (threshold>=0?double(norm<threshold):1-norm)*strength;
            values[j]=sum;
        }
    }
}
struct compression_prefix_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { compression_prefix(index,args...); }
};
template<class T> __device__ void compression_field(KernelIndex vcr_launch, const T* values,float* field,Layout g,int axis) {
    EACH_INDEX(g.size) for(int a=0;a<g.dim;++a) field[i*g.dim+a]=a==axis?-values[i]*double(g.shape[a]-1)/g.shape[a]:0;
}
struct compression_field_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { compression_field(index,args...); }
};
Parameters sample_sheet(const SheetCompression& t,const Sample& s,Random& r,bool record,Session& session) {
    auto [p,chosen,strength,sigma,smooth]=sheet_parameters(t,s,r); Grid g(p.shape); const auto d=p.shape.size();
    const auto& image=session.arrays.at(t.image);
    const auto build = [&]<class T>() {
    Buffer<T> values(g.size); View<T> data{values.ptr,g.size};
    copy_device(data,View<T>{image.view<T>().ptr,g.size});
    if(sigma>.5) blur(data,g,std::vector<double>(d,sigma),6,Boundary::Mirror,true);
    auto stats=bounds(data);
    launch(blocks(g.size/g.shape[chosen]),256,0,vcr::device::stream(), compression_prefix_launch{}, values.ptr,Layout(g),int(chosen),strength,t.threshold,stats.ptr);
    if(smooth>.5) {std::vector<double> sigmas(d,smooth);sigmas[chosen]=0;blur(data,g,sigmas,6,Boundary::Mirror,true);}
    auto field=std::make_shared<Buffer<float>>(g.size*d);
    launch(blocks(g.size),256,0,vcr::device::stream(), compression_field_launch{}, values.ptr,field->ptr,Layout(g),int(chosen));
    check(cudaGetLastError());p.device_field=field;p.border=true;
    if(record) {p.field.resize(field->size);field->download(p.field.data());}
    };
    if (image.bytes==4) build.template operator()<float>(); else build.template operator()<double>();
    return p;
}
void apply_cuda(const SheetCompression&,Sample& s,const Parameters& p,Session& session) {geometry_cuda(s,p,session);}
}
