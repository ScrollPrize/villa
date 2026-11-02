#pragma once
#include "vc_reader/resident_sampling.hpp"
#include "../sampling_math.hpp"
#include "../grid/derivatives.hpp"
#ifdef __CUDACC__
#define VCR_RES_HD __host__ __device__
#else
#define VCR_RES_HD
#endif
namespace vcr::sampling {
VCR_RES_HD inline std::size_t volumeSize(const Dense& s) {std::size_t n=1;for(int a=0;a<s.dimensions;++a)n*=s.shape[a];return n;}
struct DenseCell {
    std::int64_t lo[3]{};float frac[3]{},derivative[3]{1,1,1};bool outside=false;
};
VCR_RES_HD inline DenseCell denseCell(const Dense& s,const float* point) {
    DenseCell c;
    for(int a=0;a<s.dimensions;++a){
        float p=point[a];
        if(!std::isfinite(p)||fabsf(p)>0x1p30f){c.outside=true;return c;}
        if(s.boundary==Boundary::Constant&&(p<0||p>s.shape[a]-1)){c.outside=true;return c;}
        if(s.boundary==Boundary::Border){
            if(p<=0){p=0;c.derivative[a]=0;}else if(p>=s.shape[a]-1){p=float(s.shape[a]-1);c.derivative[a]=0;}
        }
        if(s.order==0){c.lo[a]=std::int64_t(floorf(p+.5f));c.frac[a]=0;}
        else splitCoordinate(p,0,c.lo[a],c.frac[a]);
    }
    return c;
}
VCR_RES_HD inline std::int64_t cornerIndex(const Dense& s,const DenseCell& c,int k) {
    std::int64_t at=0;
    for(int a=0;a<s.dimensions;++a){auto x=c.lo[a]+(s.order?((k>>a)&1):0);
        if(s.boundary==Boundary::Border)x=x<0?0:x>=s.shape[a]?s.shape[a]-1:x;
        if(x<0||x>=s.shape[a])return -1;
        at=at*s.shape[a]+x;
    }return at;
}
VCR_RES_HD inline float weight(const Dense& s,const DenseCell& c,int k,int derivative=-1) {
    if(!s.order)return derivative<0?1.f:0.f;
    float w=1;
    for(int a=0;a<s.dimensions;++a)w*=a==derivative?(((k>>a)&1)?1.f:-1.f)*c.derivative[a]:(((k>>a)&1)?c.frac[a]:1-c.frac[a]);
    return w;
}
VCR_RES_HD inline void denseForward(const Dense& s,float* output,std::size_t i) {
    const auto q=i%s.queries,ch=(i/s.queries)%s.channels,b=i/(s.channels*s.queries);
    auto c=denseCell(s,s.points+(b*s.queries+q)*s.dimensions);
    float result=c.outside?s.fill:0;
    if(!c.outside)for(int k=0;k<(s.order?(1<<s.dimensions):1);++k){
        auto at=cornerIndex(s,c,k);float w=weight(s,c,k);
        if(w!=0)result+=w*(at<0?s.fill:s.values[(b*s.channels+ch)*volumeSize(s)+at]);
    }
    output[i]=result;
}
template<class Add> VCR_RES_HD inline void denseBackward(const Dense& s,const float* cot,float* vg,float* pg,std::size_t query,Add add) {
    auto b=query/s.queries,q=query%s.queries;
    auto c=denseCell(s,s.points+query*s.dimensions);
    float grad[3]{};
    if(!c.outside)for(std::size_t ch=0;ch<s.channels;++ch){
        float g=cot[(b*s.channels+ch)*s.queries+q];
        for(int k=0;k<(s.order?(1<<s.dimensions):1);++k){
            auto at=cornerIndex(s,c,k);float v=at<0?s.fill:s.values[(b*s.channels+ch)*volumeSize(s)+at];
            if(vg&&at>=0)add(vg+(b*s.channels+ch)*volumeSize(s)+at,g*weight(s,c,k));
            for(int a=0;a<s.dimensions;++a)grad[a]+=g*v*weight(s,c,k,a);
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
    rc[0]=(t.rowStart+int(i/(t.colEnd-t.colStart))+.5f)*t.rows/t.outputRows;
    rc[1]=(t.colStart+int(i%(t.colEnd-t.colStart))+.5f)*t.cols/t.outputCols;
}
VCR_RES_HD inline void fieldTileForward(const float* field,const bool* valid,const Tile& t,float* out,bool* ok,std::size_t i) {
    float rc[2];tileCoordinate(t,i,rc);
    Dense s;s.values=field;s.points=rc;s.dimensions=2;s.shape[0]=t.rows;s.shape[1]=t.cols;
    s.batches=s.channels=s.queries=1;s.boundary=Boundary::Border;
    auto c=denseCell(s,rc);bool supported=true;
    float value=0;
    for(int k=0;k<4;++k){auto at=cornerIndex(s,c,k);supported=supported&&valid[at];value+=weight(s,c,k)*(valid[at]?field[at]:0.f);}
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
