#pragma once
#include "../aligned/common.hpp"
#include "../geometry/math.hpp"
#ifdef __CUDACC__
#define VCR_SURFACE_HD __host__ __device__
#else
#define VCR_SURFACE_HD
#endif
namespace vcr::surface {
using geometry::V; using geometry::dot; using geometry::cross; using geometry::vector;
using geometry::linear; using geometry::transform; using geometry::clamp; using geometry::nearest;
VCR_SURFACE_HD inline float barycentric(float a,float b,float c,float u,float v){
    // Preserve constant edge coordinates, particularly clipped half-pixel ties.
    if(u==0)return a+v*(c-a);
    if(v==0)return a+u*(b-a);
    if(u+v==1)return b+v*(c-b);
    return a+u*(b-a)+v*(c-a);
}
// Closest point on a triangle, with barycentric coordinates (Ericson regions).
VCR_SURFACE_HD inline V closest(V p,V a,V b,V c,float& u,float& v){
    V ab=b-a,ac=c-a,ap=p-a; float d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0 && d2<=0){u=v=0;return a;}
    V bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0 && d4<=d3){u=1;v=0;return b;}
    float vc=d1*d4-d3*d2;
    if(vc<=0 && d1>=0 && d3<=0){u=d1/(d1-d3);v=0;return a+ab*u;}
    V cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0 && d5<=d6){u=0;v=1;return c;}
    float vb=d5*d2-d1*d6;
    if(vb<=0 && d2>=0 && d6<=0){u=0;v=d2/(d2-d6);return a+ac*v;}
    float va=d3*d6-d5*d4;
    if(va<=0 && d4-d3>=0 && d5-d6>=0){v=(d4-d3)/((d4-d3)+(d5-d6));u=1-v;return b+(c-b)*v;}
    float denom=1/(va+vb+vc);u=vb*denom;v=vc*denom;return a+ab*u+ac*v;
}
VCR_SURFACE_HD inline float segments(V p,V q,V a,V b){
    V d=q-p,e=b-a,r=p-a;float aa=dot(d,d),bb=dot(d,e),cc=dot(e,e),dd=dot(d,r),ee=dot(e,r);
    float s=aa>0 ? clamp((bb*ee-cc*dd)/(aa*cc-bb*bb)) : 0;
    if(!(aa*cc-bb*bb>0))s=0;
    float t=cc>0?(bb*s+ee)/cc:0;
    if(t<0){t=0;s=aa>0?clamp(-dd/aa):0;}
    else if(t>1){t=1;s=aa>0?clamp((bb-dd)/aa):0;}
    V delta=r+d*s-e*t;return dot(delta,delta);
}
VCR_SURFACE_HD inline bool separated(V axis,V a,V b,V c){
    float x=dot(axis,a),y=dot(axis,b),z=dot(axis,c);
    float r=.5f*(fabsf(axis.x)+fabsf(axis.y)+fabsf(axis.z));
    return fminf(x,fminf(y,z))>r || fmaxf(x,fmaxf(y,z))< -r;
}
VCR_SURFACE_HD inline bool intersects(V a,V b,V c){
    V axes[3]{{1,0,0},{0,1,0},{0,0,1}}, edges[3]{b-a,c-b,a-c};
    for(auto axis:axes)if(separated(axis,a,b,c))return false;
    if(separated(cross(edges[0],edges[1]),a,b,c))return false;
    for(auto e:edges)for(auto axis:axes)if(separated(cross(e,axis),a,b,c))return false;
    return true;
}
// Exact feature-distance enumeration for a triangle and an affine voxel cell.
// The SAT test handles intersections; disjoint convex polyhedra attain their
// distance on a vertex/face or edge/edge pair, including band edge/corner caps.
VCR_SURFACE_HD inline bool band(V a,V b,V c,const float* metric,float radius){
    if(intersects(a,b,c))return true;
    if(radius==0)return false;
    a=linear(metric,a);b=linear(metric,b);c=linear(metric,c);
    V corners[8];for(int k=0;k<8;++k)corners[k]=linear(metric,{k&1?.5f:-.5f,k&2?.5f:-.5f,k&4?.5f:-.5f});
    float best=INFINITY,u,v;V tri[3]{a,b,c};
    for(auto p:corners){V d=p-closest(p,a,b,c,u,v);best=fminf(best,dot(d,d));}
    for(int axis=0;axis<3;++axis)for(int side=0;side<2;++side){
        int bit=1<<axis,other1=1<<((axis+1)%3),other2=1<<((axis+2)%3),k=side?bit:0;
        for(auto p:tri){
            V d=p-closest(p,corners[k],corners[k|other1],corners[k|other1|other2],u,v);best=fminf(best,dot(d,d));
            d=p-closest(p,corners[k],corners[k|other1|other2],corners[k|other2],u,v);best=fminf(best,dot(d,d));
        }
    }
    for(int k=0;k<8;++k)for(int bit=1;bit<=4;bit*=2)if(!(k&bit))
        for(int j=0;j<3;++j)best=fminf(best,segments(corners[k],corners[k|bit],tri[j],tri[(j+1)%3]));
    return best<=radius*radius;
}
struct Raster {
    int shape[3]{}, channels=1, label_h=0,label_w=0;
    std::size_t count=0;
    float metric[12]{}, uv_to_label[6]{}, origin[3]{}, radius=0;
    std::int64_t label_origin[2]{};
};
// One thread owns each output voxel. Batches update it serially, so selection
// and exact categorical conflict tests never require floating-point atomics.
template<class T> VCR_SURFACE_HD void raster(const Raster& g,const float* triangles,const std::int64_t* ids,
    const std::int64_t* offsets,const int* candidates,const T* label,T* out,bool* coverage,bool* conflict,
    float* distance,std::int64_t* winner,float* uv,bool* geometry,std::size_t i){
    V center{float(i%g.shape[2])+g.origin[0],float(i/g.shape[2]%g.shape[1])+g.origin[1],float(i/g.shape[2]/g.shape[1])+g.origin[2]};
    for(auto pos=offsets[i];pos<offsets[i+1];++pos){
        int t=candidates[pos];const float* p=triangles+15*t;
        V a=vector(p)-center,b=vector(p+5)-center,c=vector(p+10)-center;
        if(!band(a,b,c,g.metric,g.radius))continue;
        geometry[i]=true;
        float u,v;V q=closest({0,0,0},linear(g.metric,a),linear(g.metric,b),linear(g.metric,c),u,v);
        float x=barycentric(p[3],p[8],p[13],u,v);
        float y=barycentric(p[4],p[9],p[14],u,v);
        float px=g.uv_to_label[0]*x+g.uv_to_label[1]*y+g.uv_to_label[2];
        float py=g.uv_to_label[3]*x+g.uv_to_label[4]*y+g.uv_to_label[5];
        if(!std::isfinite(px)||!std::isfinite(py)||fabsf(px)>0x1p30f||fabsf(py)>0x1p30f)continue;
        auto ix=nearest(px,g.label_origin[1]),iy=nearest(py,g.label_origin[0]);
        ix-=g.label_origin[1];iy-=g.label_origin[0];
        if(ix<0||iy<0||ix>=g.label_w||iy>=g.label_h)continue;
        auto pixel=iy*g.label_w+ix;float d=dot(q,q);
        bool choose=!coverage[i]||d<distance[i]||(d==distance[i]&&ids[t]<winner[i]);
        for(int ch=0;ch<g.channels;++ch){
            T value=label[std::size_t(ch)*g.label_h*g.label_w+pixel];
            if(coverage[i]&&out[ch*g.count+i]!=value)conflict[i]=true;
            if(choose)out[ch*g.count+i]=value;
        }
        if(choose){distance[i]=d;winner[i]=ids[t];uv[2*i]=x;uv[2*i+1]=y;}
        coverage[i]=true;
    }
}
struct Interpolation {
    int height=0,width=0; float physical[12]{},destination[12]{},orientation=1,offset=0;
};
VCR_SURFACE_HD inline void interpolate(const Interpolation& g,const float* xyz,const bool* mask,
    const float* queries,const float* normals,float* output,bool* valid,std::size_t i){
    float x=queries[2*i],y=queries[2*i+1];valid[i]=false;
    for(int a=0;a<3;++a)output[3*i+a]=NAN;
    if(!std::isfinite(x)||!std::isfinite(y)||x<0||y<0||x>g.width-1||y>g.height-1||g.width<2||g.height<2)return;
    int col=int(fminf(floorf(x),g.width-2)),row=int(fminf(floorf(y),g.height-2));
    float u=x-col,v=y-row;int at[4]{row*g.width+col,row*g.width+col+1,(row+1)*g.width+col,(row+1)*g.width+col+1};
    V p[4];for(int k=0;k<4;++k){if(!mask[at[k]])return;p[k]=vector(xyz+3*at[k]);}
    V point=(p[0]*(1-u)+p[1]*u)*(1-v)+(p[2]*(1-u)+p[3]*u)*v;
    V du=linear(g.physical,(p[1]-p[0])*(1-v)+(p[3]-p[2])*v);
    V dv=linear(g.physical,(p[2]-p[0])*(1-u)+(p[3]-p[1])*u);
    V normal=cross(du,dv)*g.orientation;
    if(normals){
        V n[4];for(int k=0;k<4;++k)n[k]=vector(normals+3*at[k]);
        normal=(n[0]*(1-u)+n[1]*u)*(1-v)+(n[2]*(1-u)+n[3]*u)*v;
    }
    float scale=fmaxf(fabsf(normal.x),fmaxf(fabsf(normal.y),fabsf(normal.z)));
    if(!(scale>0)||!std::isfinite(normal.x)||!std::isfinite(normal.y)||!std::isfinite(normal.z))return;
    normal=normal*(1/scale);normal=normal*(1/sqrtf(dot(normal,normal)));
    point=transform(g.destination,transform(g.physical,point)+normal*g.offset);
    if(!std::isfinite(point.x)||!std::isfinite(point.y)||!std::isfinite(point.z))return;
    output[3*i]=point.x;output[3*i+1]=point.y;output[3*i+2]=point.z;valid[i]=true;
}
}
#undef VCR_SURFACE_HD
