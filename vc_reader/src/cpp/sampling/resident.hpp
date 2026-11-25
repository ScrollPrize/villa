#pragma once
#include "vc_reader/resident_sampling.hpp"
#include "../sample_point.hpp"
#include "../grid/derivatives.hpp"
#ifdef __CUDACC__
#define VCR_RES_HD __host__ __device__
#else
#define VCR_RES_HD
#endif
namespace vcr::sampling {
VCR_RES_HD inline std::size_t volumeSize(const Dense& s) {std::size_t n=1;for(int a=0;a<s.dimensions;++a)n*=s.shape[a];return n;}
// A contiguous channel is one block of the same voxel lookup used by reader sampling.
struct DenseLookup {
    const float* values;
    std::int64_t shape[3]{1,1,1}, block[3]{1,1,1}, stride[3]{4,4,4};
    float fill;
    VCR_RES_HD DenseLookup(const Dense& s, std::size_t channel): values(s.values+channel*volumeSize(s)), fill(s.fill) {
        for(int a=0;a<s.dimensions;++a)shape[a]=block[a]=s.shape[a];
        for(int a=1;a>=0;--a)stride[a]=stride[a+1]*shape[a+1];
    }
    VCR_RES_HD float load(const unsigned char* p)const{return *reinterpret_cast<const float*>(p);}
    VCR_RES_HD const unsigned char* voxel(const std::int64_t* v,bool&)const{
        return reinterpret_cast<const unsigned char*>(values)+v[0]*stride[0]+v[1]*stride[1]+v[2]*stride[2];
    }
};
// Boundary policy and tensor layout are adapters; interpolation belongs to point().
VCR_RES_HD inline bool denseCoordinates(const Dense& s,const float* input,float* q,float* derivative) {
    for(int a=0;a<s.dimensions;++a){
        float p=input[a];
        if(!std::isfinite(p)||fabsf(p)>0x1p30f)return false;
        if(s.boundary==Boundary::Constant&&(p<0||p>s.shape[a]-1))return false;
        derivative[a]=1;
        if(s.boundary==Boundary::Border){
            if(p<=0){p=0;derivative[a]=0;}else if(p>=s.shape[a]-1){p=float(s.shape[a]-1);derivative[a]=0;}
        }
        q[a]=p;
    }
    return true;
}
VCR_RES_HD inline void denseForward(const Dense& s,float* output,std::size_t i) {
    const auto query=i%s.queries,ch=(i/s.queries)%s.channels,b=i/(s.channels*s.queries);
    float q[3]{},derivative[3]{};output[i]=s.fill;
    if(!denseCoordinates(s,s.points+(b*s.queries+query)*s.dimensions,q,derivative))return;
    DenseLookup lookup(s,b*s.channels+ch);const std::int64_t base[3]{};
    point(lookup,base,q,s.order,output[i]);
}
template<class Add> VCR_RES_HD inline void denseBackward(const Dense& s,const float* cot,float* vg,float* pg,std::size_t query,Add add) {
    const auto b=query/s.queries,qindex=query%s.queries;
    float q[3]{},derivative[3]{},grad[3]{};
    if(denseCoordinates(s,s.points+query*s.dimensions,q,derivative)){
        PointCell cell;
        // All axes participate in a derivative, including at integer coordinates.
        cellOf(q,s.dimensions,cell,[](std::size_t){return 0;},true);
        if(!s.order){for(int a=0;a<s.dimensions;++a)cell.lo[a]+=nearest_rounds_up(cell.frac[a]);cell.k=0;}
        for(std::size_t ch=0;ch<s.channels;++ch){
            const auto channel=(b*s.channels+ch)*volumeSize(s);
            const float g=cot[(b*s.channels+ch)*s.queries+qindex];
            for(std::size_t k=0;k<(std::size_t(1)<<cell.k);++k){
                std::int64_t at=0;bool inside=true;
                for(int a=0;a<s.dimensions;++a){
                    auto x=cell.lo[a]+(s.order?int((k>>a)&1):0);
                    if(s.boundary==Boundary::Border)x=x<0?0:x>=s.shape[a]?s.shape[a]-1:x;
                    inside&=x>=0&&x<s.shape[a];at=at*s.shape[a]+x;
                }
                const float v=inside?s.values[channel+at]:s.fill;
                if(vg&&inside)add(vg+channel+at,g*cornerWeight(k,cell.frac,cell.axes,cell.k));
                if(s.order)for(int a=0;a<s.dimensions;++a)
                    grad[a]+=g*v*cornerWeight(k,cell.frac,cell.axes,cell.k,a)*derivative[a];
            }
        }
    }
    if(pg)for(int a=0;a<s.dimensions;++a)pg[query*s.dimensions+a]=grad[a];
}
VCR_RES_HD inline grid::Derivatives surfaceView(const float* xyz,const bool* valid,float* derivatives,int batch,int h,int w){
    std::size_t n=std::size_t(h)*w;
    grid::GridView view;view.xyz=xyz+batch*n*3;view.valid=valid+batch*n;
    if(derivatives){view.du=derivatives+batch*n*9;view.dv=view.du+n*3;view.duv=view.dv+n*3;}
    return {view,h,w};
}
VCR_RES_HD inline void surfaceForward(const float* xyz,const bool* valid,const float* derivatives,int h,int w,
 const float* rc,std::size_t queries,float* p,float* row,float* col,bool* ok,int mode,bool border,std::size_t i){
    auto d=surfaceView(xyz,valid,const_cast<float*>(derivatives),int(i/queries),h,w);
    grid::PointMap map;map.height=h;map.width=w;
    if(mode!=2){d.v.du=nullptr;d.v.dv=nullptr;d.v.duv=nullptr;}
    d.v.triangles=mode==3;
    float r=rc[i*2],c=rc[i*2+1];
    if(border && std::isfinite(r) && std::isfinite(c)){r=fminf(fmaxf(r,0),h-1);c=fminf(fmaxf(c,0),w-1);}
    float out[3],du[3],dv[3];
    bool good=grid::point(map,d.v,c,r,out,nullptr,du,dv);
    ok[i]=good;
    for(int a=0;a<3;++a){p[i*3+a]=good?out[a]:0;row[i*3+a]=good?dv[a]:0;col[i*3+a]=good?du[a]:0;}
}
VCR_RES_HD inline void tileCoordinate(const Tile& t,std::size_t i,float* rc) {
    rc[0]=pixelCenter(t.rowStart+int(i/(t.colEnd-t.colStart)),t.rows,t.outputRows);
    rc[1]=pixelCenter(t.colStart+int(i%(t.colEnd-t.colStart)),t.cols,t.outputCols);
}
VCR_RES_HD inline void fieldTileForward(const float* field,const bool* valid,const Tile& t,float* out,bool* ok,std::size_t i) {
    float rc[2];tileCoordinate(t,i,rc);
    const int shape[2]{t.rows,t.cols};bool supported=true;float value=0;
    clampedCorners<2>(shape,rc,[&](std::size_t at,float weight){
        supported=supported&&valid[at];value+=weight*(valid[at]?field[at]:0.f);
    });
    out[i]=value;ok[i]=supported&&std::isfinite(value);
}
VCR_RES_HD inline void surfaceTileForward(const float* xyz,const bool* valid,const float* derivatives,const Tile& t,int mode,float* out,bool* ok,std::size_t i) {
    float rc[2],p[3],row[3],col[3];tileCoordinate(t,i,rc);
    surfaceForward(xyz,valid,derivatives,t.rows,t.cols,rc,1,p,row,col,ok+i,mode,true,0);out[i]=p[0];
}
#ifdef VC_READER_CUDA
void fieldTileCuda(const float*,const bool*,const Tile&,float*,bool*,int);
void surfaceTileCuda(const float*,const bool*,const float*,const Tile&,int,float*,bool*,int);
void denseCuda(const Dense&,float*,int);
void denseBackwardCuda(const Dense&,const float*,float*,float*,int);
void prepareSurfacesCuda(const float*,const bool*,int,int,int,float*,int);
void sampleSurfacesCuda(const float*,const bool*,const float*,int,int,int,const float*,std::size_t,float*,float*,float*,bool*,int,bool,int);
#endif
}
#undef VCR_RES_HD
