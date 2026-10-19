#pragma once
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define VCR_GEOMETRY_HD __host__ __device__
#else
#define VCR_GEOMETRY_HD
#endif
namespace vcr::geometry {
struct V { float x,y,z;
    VCR_GEOMETRY_HD V operator+(V b)const{return{x+b.x,y+b.y,z+b.z};}
    VCR_GEOMETRY_HD V operator-(V b)const{return{x-b.x,y-b.y,z-b.z};}
    VCR_GEOMETRY_HD V operator*(float s)const{return{x*s,y*s,z*s};}
};
VCR_GEOMETRY_HD inline float dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
VCR_GEOMETRY_HD inline V cross(V a,V b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
VCR_GEOMETRY_HD inline V vector(const float* p){return{p[0],p[1],p[2]};}
VCR_GEOMETRY_HD inline void store(float* p,V v){p[0]=v.x;p[1]=v.y;p[2]=v.z;}
VCR_GEOMETRY_HD inline V linear(const float* m,V p){return{dot(vector(m),p),dot(vector(m+4),p),dot(vector(m+8),p)};}
VCR_GEOMETRY_HD inline V transform(const float* m,V p){return linear(m,p)+V{m[3],m[7],m[11]};}
VCR_GEOMETRY_HD inline float clamp(float v){return fminf(1,fmaxf(0,v));}
VCR_GEOMETRY_HD inline bool finite(V v){return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);}
VCR_GEOMETRY_HD inline V closest_segment(V p,V a,V b,float& t){V d=b-a;float n=dot(d,d);t=n>0?clamp(dot(p-a,d)/n):0;return a+d*t;}
VCR_GEOMETRY_HD inline std::int64_t nearest(float q,std::int64_t base){
    float f=floorf(q);auto n=base+std::int64_t(f);float r=q-f;
    return n+(r>.5f || (r==.5f && n%2!=0));
}
}
