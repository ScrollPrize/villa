#pragma once
#include "math.hpp"
#include "../rounding.hpp"
#include <cstdint>
namespace vcr::geometry {
struct MeshSlice {
    const float* vertices=nullptr;
    const std::int32_t* triangles=nullptr;
    const std::uint16_t* labels=nullptr;
    std::size_t vertex_count=0, triangle_count=0;
    const float *vertex_normals=nullptr, *face_normals=nullptr;
    const float *vertex_tangents=nullptr, *vertex_bitangents=nullptr, *face_tangents=nullptr, *face_bitangents=nullptr;
    const bool *vertex_normal_flags=nullptr, *vertex_tangent_flags=nullptr;
};
struct SliceOutput {
    int width=0,height=0;
    std::uint16_t* labels=nullptr;
    float *normals=nullptr, *frames=nullptr;
    float *normal_sums=nullptr,*tangent_sums=nullptr,*bitangent_sums=nullptr;
    std::uint32_t* counts=nullptr;
};
// Ordered DDA writes: later triangles win labels; vector sums follow triangle order.
// row>=0 restricts writes to one independent scanline for deterministic CUDA execution.
VCR_GEOMETRY_HD inline int raster_triangle(const MeshSlice& m, std::size_t face, double z, SliceOutput out, int row=-1) {
    const auto* ids=m.triangles+face*3;
    const float* v[3];
    for(int k=0;k<3;++k) {
        if(ids[k]<0 || std::size_t(ids[k])>=m.vertex_count)return 1;
        v[k]=m.vertices+std::size_t(ids[k])*3;
        for(int j=0;j<3;++j)if(!std::isfinite(v[k][j]) || fabsf(v[k][j])>1e9f)return 1;
    }
    if(z<fminf(v[0][2],fminf(v[1][2],v[2][2])) || z>fmaxf(v[0][2],fmaxf(v[1][2],v[2][2])))return 0;
    if(row>=0 && (row<fminf(v[0][1],fminf(v[1][1],v[2][1]))-.5 || row>fmaxf(v[0][1],fmaxf(v[1][1],v[2][1]))+.5))return 0;
    float points[3][2]{}, bary[3][3]{};int count=0;
    for(int a=0;a<3;++a) {
        int b=(a+1)%3;double t=0;
        if(fabs(double(v[a][2])-z)<1e-8)t=0;
        else if(fabs(double(v[b][2])-z)<1e-8)t=1;
        else {
            const double delta=double(v[b][2])-v[a][2];if(fabs(delta)<1e-15)continue;
            t=(z-v[a][2])/delta;if(t<-.001 || t>1.001)continue;
        }
        float x=float(v[a][0]+t*(double(v[b][0])-v[a][0])),y=float(v[a][1]+t*(double(v[b][1])-v[a][1]));
        bool duplicate=false;
        for(int j=0;j<count;++j){double dx=double(x)-points[j][0],dy=double(y)-points[j][1];if(dx*dx+dy*dy<1e-12)duplicate=true;}
        if(duplicate)continue;
        points[count][0]=x;points[count][1]=y;bary[count][a]=float(1-t);bary[count][b]=float(t);++count;
    }
    for(int a=0;a<count;++a)for(int b=a+1;b<count;++b) {
        const float dx=points[b][0]-points[a][0],dy=points[b][1]-points[a][1];
        const auto steps=std::int64_t(fmaxf(fabsf(dx),fabsf(dy)));
        const double sx=steps?double(dx)/steps:0,sy=steps?double(dy)/steps:0;
        double x=points[a][0],y=points[a][1];
        for(std::int64_t step=0;step<=steps;++step,x+=sx,y+=sy) {
            const auto ix=std::int64_t(vcr::round_half_even(x)),iy=std::int64_t(vcr::round_half_even(y));
            if(ix<0 || ix>=out.width || iy<0 || iy>=out.height || (row>=0 && iy!=row))continue;
            const std::size_t pixel=std::size_t(iy)*out.width+ix;out.labels[pixel]=m.labels[face];
            if(!out.normals)continue;
            const double alpha=steps?double(step)/steps:0;
            double weights[3];for(int k=0;k<3;++k)weights[k]=bary[a][k]*(1-alpha)+bary[b][k]*alpha;
            for(int j=0;j<3;++j) {
                double n=0,tan=0,bit=0;
                for(int k=0;k<3;++k) {
                    n+=weights[k]*(m.vertex_normal_flags[face]?m.vertex_normals[std::size_t(ids[k])*3+j]:m.face_normals[face*3+j]);
                    if(out.frames) {
                        tan+=weights[k]*(m.vertex_tangent_flags[face]?m.vertex_tangents[std::size_t(ids[k])*3+j]:m.face_tangents[face*3+j]);
                        bit+=weights[k]*(m.vertex_tangent_flags[face]?m.vertex_bitangents[std::size_t(ids[k])*3+j]:m.face_bitangents[face*3+j]);
                    }
                }
                out.normal_sums[pixel*3+j]+=float(n);
                if(out.frames){out.tangent_sums[pixel*3+j]+=float(tan);out.bitangent_sums[pixel*3+j]+=float(bit);}
            }
            ++out.counts[pixel];
        }
    }
    return 0;
}
VCR_GEOMETRY_HD inline int finish_slice_pixel(SliceOutput out,std::size_t i) {
    if(!out.normals || !out.counts[i])return 0;
    V n=vector(out.normal_sums+3*i)*(1.f/out.counts[i]);float length=sqrtf(dot(n,n));
    if(!(length>1e-12f))return 2;n=n*(1/length);store(out.normals+3*i,n);
    if(!out.frames)return 0;
    V t=vector(out.tangent_sums+3*i)*(1.f/out.counts[i]);t=t-n*dot(n,t);length=sqrtf(dot(t,t));
    if(!(length>1e-12f))return 3;t=t*(1/length);
    V b={n.y*t.z-n.z*t.y,n.z*t.x-n.x*t.z,n.x*t.y-n.y*t.x};
    if(dot(b,vector(out.bitangent_sums+3*i))<0)b=b*(-1.f);
    length=sqrtf(dot(b,b));if(!(length>1e-12f))return 4;b=b*(1/length);
    store(out.frames+9*i,t);store(out.frames+9*i+3,b);store(out.frames+9*i+6,n);return 0;
}
}
