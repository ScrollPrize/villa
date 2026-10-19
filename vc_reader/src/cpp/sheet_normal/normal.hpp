#pragma once
#include "../symmetric_eigen.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#ifdef __CUDACC__
#define VCR_SHEET_HD __host__ __device__
#else
#define VCR_SHEET_HD
#endif
namespace vcr::sheet_normal {
constexpr int width=109, sites=33, count=sites*sites*sites;
constexpr int zcount=sites*width*width, ycount=sites*sites*width;
struct Kernels {
    float taps[4][13]{}; // smooth/derivative sigma 2, then sigma 1, zero padded
    Kernels() {
        for(int scale=0;scale<2;++scale) {
            double sigma=scale ? 1. : 2., sum=0;
            int r=scale ? 3 : 6;
            for(int j=-r;j<=r;++j) sum+=std::exp(-.5*j*j/(sigma*sigma));
            for(int j=-r;j<=r;++j) {
                double g=std::exp(-.5*j*j/(sigma*sigma))/sum;
                taps[scale*2][j+6]=float(g);
                taps[scale*2+1][j+6]=float(g*j/(sigma*sigma));
            }
        }
    }
};
// Only the seven tensor sums and the 3x3 eigensystem use float64: the
// relative eigengap validity threshold is 1e-6, including low-contrast CT.
// Images, separable filters, gradients and magnitudes are float32.
VCR_SHEET_HD inline bool principal(const double* sums, double* normal, double& gap) {
    if(!(sums[6]>0)) return false;
    double a[9]={sums[0],sums[1],sums[2],sums[1],sums[3],sums[4],sums[2],sums[4],sums[5]};
    for(double& v:a) v/=sums[6];
    double values[3], vectors[9];
    math::symmetric_eigen(a,3,values,vectors);
    if(!(values[2]>1e-12)) return false;
    gap=(values[2]-values[1])/values[2];
    if(!(gap>1e-6)) return false;
    for(int j=0;j<3;++j) normal[j]=vectors[3*j+2];
    return true;
}
VCR_SHEET_HD inline float magnitude(float x,float y,float z) {
    return float(sqrt(double(x)*x+double(y)*y+double(z)*z));
}
VCR_SHEET_HD inline void contribution(int i,const float* g,const float* center,
                                    const double* normal,float floor,bool refine,double* sums) {
    double x=3*(i%sites)-center[0], y=3*((i/sites)%sites)-center[1], z=3*(i/(sites*sites))-center[2];
    double exponent=(x*x+y*y+z*z)/576.;
    if(refine) { double along=x*normal[0]+y*normal[1]+z*normal[2]; exponent+=along*along*(1./16.-1./576.); }
    double gx=g[i], gy=g[count+i], gz=g[2*count+i];
    double weight=exp(-.5*exponent);
    if(refine) { float mag=magnitude(float(gx),float(gy),float(gz)); weight/=mag>floor ? mag : floor; }
    sums[0]+=weight*gx*gx; sums[1]+=weight*gx*gy; sums[2]+=weight*gx*gz;
    sums[3]+=weight*gy*gy; sums[4]+=weight*gy*gz; sums[5]+=weight*gz*gz; sums[6]+=weight;
}
template<class T> inline void cpu(const T* image,const float* centers,float* output,std::size_t batch) {
    Kernels k;
    auto z=std::make_unique_for_overwrite<float[]>(4*zcount);
    auto y=std::make_unique_for_overwrite<float[]>(6*ycount);
    auto g=std::make_unique_for_overwrite<float[]>(6*count);
    std::vector<float> mags(count);
    for(std::size_t b=0;b<batch;++b) {
        std::fill_n(output+b*4,4,0.f);
        if(!centers[4*b+3]) continue;
        const T* src=image+b*width*width*width;
        for(int c=0;c<4;++c) for(int zz=0;zz<sites;++zz) {
            float* dst=z.get()+c*zcount+zz*width*width;
            std::fill_n(dst,width*width,0.f);
            for(int j=0;j<13;++j) {
                const T* plane=src+(3*zz+j)*width*width;
                const float w=k.taps[c][j];
                for(int i=0;i<width*width;++i) dst[i]+=w*(float(plane[i])*(1.f/255.f));
            }
        }
        for(int c=0;c<6;++c) for(int zz=0;zz<sites;++zz) for(int yy=0;yy<sites;++yy) {
            int scale=c/3, axis=c%3, zc=2*scale+(axis==2), yc=2*scale+(axis==1);
            float* dst=y.get()+c*ycount+(zz*sites+yy)*width;
            std::fill_n(dst,width,0.f);
            for(int j=0;j<13;++j) {
                const float* row=z.get()+zc*zcount+(zz*width+3*yy+j)*width;
                for(int xx=0;xx<width;++xx) dst[xx]+=k.taps[yc][j]*row[xx];
            }
        }
        for(int c=0;c<6;++c) for(int i=0;i<count;++i) {
            int xc=2*(c/3)+(c%3==0), x=i%sites, row=i/sites;
            float value=0;
            for(int j=0;j<13;++j) value+=k.taps[xc][j]*y[c*ycount+row*width+3*x+j];
            g[c*count+i]=value;
        }
        for(int i=0;i<count;++i) mags[i]=magnitude(g[3*count+i],g[4*count+i],g[5*count+i]);
        std::nth_element(mags.begin(),mags.begin()+count/2,mags.end());
        float floor=std::max(mags[count/2],1e-8f);
        double normal[3]{},gap=0; bool valid=true;
        for(int pass=0;pass<4 && valid;++pass) {
            double sums[7]{};
            for(int i=0;i<count;++i) contribution(i,g.get()+(pass ? 3*count : 0),centers+b*4,normal,floor,pass!=0,sums);
            valid=principal(sums,normal,gap);
        }
        if(valid) { for(int j=0;j<3;++j) output[4*b+j]=float(normal[j]); output[4*b+3]=float(gap); }
    }
}
}
#undef VCR_SHEET_HD
