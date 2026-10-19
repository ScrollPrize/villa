#include "../tracing.hpp"
#include "geometry.cuh"
namespace vcr::augment::cuda {
__device__ void strip_offsets(KernelIndex vcr_launch, float* field,Layout g,const double* controls,std::size_t count,double stride,double a,double b,double offset) {
    EACH_INDEX(g.size) {
        const double x=offset+a*double(g.coord(i,0))+b*double(g.coord(i,1));
        field[i*2]=cubic(controls,count,x/stride+1); field[i*2+1]=0;
    }
}
struct strip_offsets_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { strip_offsets(index,args...); }
};
Parameters sample_strip(const StripGeometry& t,const Sample& s,Random& r,bool record) {
    std::vector<double> controls; double stride; auto p=strip_parameters(t,s,r,controls,stride);
    if (!controls.empty()) {
        Buffer<double> knots(controls); auto field=std::make_shared<Buffer<float>>(product(p.shape)*2);
        launch(blocks(field->size/2),256,0,vcr::device::stream(), strip_offsets_launch{}, field->ptr,Layout{Grid(p.shape)},knots.ptr,knots.size,stride,p.matrix[2],p.matrix[3],p.offset[1]);
        check(cudaGetLastError()); p.device_field=field;
        if(record) {p.field.resize(field->size);field->download(p.field.data());}
    }
    return p;
}
void apply_cuda(const StripGeometry&,Sample& s,const Parameters& p,Session& session) {geometry_cuda(s,p,session);}
}
