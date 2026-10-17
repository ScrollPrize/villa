#pragma once
#include "spatial.hpp"
#include "volume_math.hpp"

namespace vcr::augment {
// Related public transforms share a flat configuration and execution machinery.
// Public names remain separate constructors; kinds are typed, never strings.
enum class VolumeKind { BiasField, PlasmaBrightnessContrast, PlasmaShadow, Illumination, Vignetting,
    RandomToneCurve, Equalize, CLAHE, HistogramMatching, PixelDistributionAdaptation, FDA,
    MultiplicativeNoise, ShotNoise, AdditiveNoise, SaltAndPepper, MotionBlur, Defocus,
    ZoomBlur, UnsharpMask, GibbsRinging, RingingOvershoot };
struct VolumeIntensity {
    VolumeKind kind=VolumeKind::BiasField;
    Scalar amount=.2, secondary=.2, scale=.05, extra=1;
    ChannelOptions options;
    int mode=0, bins=256, octaves=5, samples=9;
    Shape tile{16,16,16}, reference_shape;
    std::vector<double> reference;
    std::vector<double> direction, center;
    bool shared_noise=false;
};
namespace volume_detail {
struct Effect {
    VolumeKind kind;
    double a,b,c,d, axis[3],center[3];
    std::uint64_t seed;
    int mode,bins,octaves,samples;
    std::size_t tiles[3],tile_count;
    Effect(const VolumeIntensity& t,const ChannelPlan& p,const Grid3& g):kind(t.kind),
        a(p.values[0]),b(p.values[1]),c(p.values[2]),d(p.values[3]),seed(p.seed),
        mode(t.mode),bins(t.bins),octaves(t.octaves),samples(t.samples),tile_count(1) {
        for(int j=0;j<3;++j){axis[j]=p.values[4+j];center[j]=p.values[7+j];tiles[j]=(g.n[j]+t.tile[j]-1)/t.tile[j];tile_count*=tiles[j];}
    }
};
inline bool histogram_kind(VolumeKind k) {return k==VolumeKind::Equalize || k==VolumeKind::CLAHE || k==VolumeKind::HistogramMatching;}
inline bool frequency_kind(VolumeKind k) {return k==VolumeKind::FDA || k==VolumeKind::GibbsRinging;}
VCR_HD inline double fractal(const Effect& e,const Grid3& g,std::size_t i) {
    double sum=0,weight=1,total=0,freq=e.c;
    for(int o=0;o<e.octaves;++o){double q[3];for(int a=0;a<3;++a)q[a]=double(g.coord(i,a))*freq;
        sum+=weight*lattice(q,mix(e.seed+o),false,true);total+=weight;weight*=e.d;freq*=2;}
    return total?sum/total:0;
}
VCR_HD inline double equalized(const double* h,int bins,int b) {
    double first=0;for(int j=0;j<bins;++j)if(h[j]>0){first=h[j];break;}
    return first<1?clamp((h[b]-first)/(1-first),0,1):0;
}
template<class T> VCR_HD void prepare_histogram(const Effect& e,const Grid3& g,const T* v,const Stats& s,double* table,std::size_t tile) {
    double* h=table+tile*e.bins;for(int b=0;b<e.bins;++b)h[b]=0;
    if(e.kind!=VolumeKind::CLAHE){histogram(v,g.size,h,e.bins,s.low,s.high);cdf(h,e.bins);return;}
    std::size_t pos[3]{tile/(e.tiles[1]*e.tiles[2]),tile/e.tiles[2]%e.tiles[1],tile%e.tiles[2]},lo[3],hi[3];
    for(int a=0;a<3;++a){lo[a]=pos[a]*g.n[a]/e.tiles[a];hi[a]=(pos[a]+1)*g.n[a]/e.tiles[a];}
    for(std::size_t z=lo[0];z<hi[0];++z)for(std::size_t y=lo[1];y<hi[1];++y)for(std::size_t x=lo[2];x<hi[2];++x)
        h[bin(v[z*g.stride[0]+y*g.stride[1]+x],s.low,s.high,e.bins)]+=1;
    cdf(h,e.bins,e.a);
}
VCR_HD inline double clahe_value(const Effect& e,const Grid3& g,std::size_t i,int b,const double* table) {
    std::size_t lo[3],hi[3];double f[3];
    for(int a=0;a<3;++a){double q=clamp((double(g.coord(i,a))+.5)*e.tiles[a]/g.n[a]-.5,0,double(e.tiles[a]-1));lo[a]=std::size_t(floor(q));hi[a]=lo[a]+1<e.tiles[a]?lo[a]+1:lo[a];f[a]=q-lo[a];}
    double result=0;
    for(int corner=0;corner<8;++corner){std::size_t t=0;double w=1;for(int a=0;a<3;++a){bool bit=(corner>>a)&1;t=t*e.tiles[a]+(bit?hi[a]:lo[a]);w*=bit?f[a]:1-f[a];} result+=w*table[t*e.bins+b];}
    return result;
}
// One voxel, reading immutable input and shared summaries. All spatial kernels
// use Z/Y/X jointly and nearest boundary extension, including singleton axes.
template<class T> VCR_HD double effect_value(const Effect& e,const Grid3& g,const T* v,std::size_t i,
                                              const Stats& s,const double* table,const Stats& ref) {
    double x=v[i],span=s.high-s.low,q[3];for(int a=0;a<3;++a)q[a]=double(g.coord(i,a));
    switch(e.kind) {
    case VolumeKind::BiasField: {
        for(int a=0;a<3;++a)q[a]*=e.c;return x*exp(e.a*lattice(q,e.seed,true,false)); }
    case VolumeKind::PlasmaBrightnessContrast: {double f=fractal(e,g,i);return (x-s.mean)*(1+e.b*f)+s.mean+e.a*span*f;}
    case VolumeKind::PlasmaShadow: return x*(1-e.a*(fractal(e,g,i)+1)*.5);
    case VolumeKind::Illumination: {
        double field=0;
        if(e.mode==0){double denom=0;for(int a=0;a<3;++a){field+=e.axis[a]*(g.n[a]>1?2*q[a]/double(g.n[a]-1)-1:0);denom+=fabs(e.axis[a]);}field/=denom;}
        else {double dist=0;for(int a=0;a<3;++a){double d=(g.n[a]>1?q[a]/double(g.n[a]-1):.5)-e.center[a];dist+=d*d;}
            field=e.mode==1?1-sqrt(dist/3):exp(-dist/(2*e.c*e.c));}
        return x*(1+e.a*field); }
    case VolumeKind::Vignetting: {
        double dist=0;for(int a=0;a<3;++a){double d=(g.n[a]>1?q[a]/double(g.n[a]-1):.5)-e.center[a];dist+=d*d;}
        return x*(1-e.a*clamp(dist/(e.c*e.c),0,1)); }
    case VolumeKind::RandomToneCurve: {if(span==0)return x;double t=clamp((x-s.low)/span,0,1),u=1-t;return s.low+span*(3*u*u*t*e.a+3*u*t*t*e.b+t*t*t);}
    case VolumeKind::Equalize: return span==0?x:s.low+span*equalized(table,e.bins,bin(x,s.low,s.high,e.bins));
    case VolumeKind::CLAHE: return span==0?x:s.low+span*clahe_value(e,g,i,bin(x,s.low,s.high,e.bins),table);
    case VolumeKind::HistogramMatching: {
        double cdf_x=table[bin(x,s.low,s.high,e.bins)];const double* rh=table+e.bins;int j=0;while(j+1<e.bins && rh[j]<cdf_x)++j;
        double mapped=ref.low+(ref.high-ref.low)*double(j)/(e.bins-1);return (1-e.a)*x+e.a*mapped; }
    case VolumeKind::PixelDistributionAdaptation: {
        double mapped=e.mode==1?(span>0?(x-s.low)/span:0)*(ref.high-ref.low)+ref.low:(s.stddev>0?(x-s.mean)/s.stddev:0)*ref.stddev+ref.mean;
        return (1-e.a)*x+e.a*mapped; }
    case VolumeKind::MultiplicativeNoise: return x*(e.a+(e.b-e.a)*uniform(e.seed,i));
    case VolumeKind::ShotNoise: {Rng r{mix(e.seed+i)};return x<0?NAN:poisson(x*e.a,r)/e.a;}
    case VolumeKind::AdditiveNoise: {
        Rng r{mix(e.seed+i)};double u=r.u(),noise=0;
        if(e.mode==0)noise=(2*u-1)*e.a;
        else if(e.mode==1)noise=e.a*r.gaussian();
        else if(e.mode==2)noise=-e.a*(u<.5?-1:1)*log(1-2*fabs(u-.5));
        else {double a=gamma_sample(e.c,r),b=gamma_sample(e.d,r);noise=e.a*(a/(a+b)-e.c/(e.c+e.d));}
        return x+e.b+noise; }
    case VolumeKind::SaltAndPepper: {double u=uniform(e.seed,i);return u<e.a?(uniform(e.seed,i+g.size)<e.b?s.high:s.low):x;}
    case VolumeKind::MotionBlur: {
        if(e.a==0)return x;double sum=0;for(int j=0;j<e.samples;++j){double pos[3],t=e.a*(double(j)/(e.samples-1)-.5);for(int a=0;a<3;++a)pos[a]=q[a]+e.axis[a]*t;sum+=interpolate(v,g,pos);}return sum/e.samples; }
    case VolumeKind::Defocus: {
        if(e.a==0)return x;int radius=int(ceil(e.a));double sum=0,count=0;
        for(int z=-radius;z<=radius;++z)for(int y=-radius;y<=radius;++y)for(int x=-radius;x<=radius;++x)if(double(z*z+y*y+x*x)<=e.a*e.a){double pos[3]{q[0]+z,q[1]+y,q[2]+x};sum+=interpolate(v,g,pos);++count;}return sum/count; }
    case VolumeKind::ZoomBlur: {
        if(e.a==0)return x;double sum=0;for(int j=0;j<e.samples;++j){double pos[3],factor=1+e.a*double(j)/(e.samples-1);for(int a=0;a<3;++a){double center=e.center[a]*(g.n[a]-1);pos[a]=center+(q[a]-center)/factor;}sum+=interpolate(v,g,pos);}return sum/e.samples; }
    case VolumeKind::UnsharpMask: return x+e.a*(x-table[i]);
    case VolumeKind::RingingOvershoot: {
        if(e.a==1)return x;int radius=int(e.b);double sum=0,weights=0;
        // Separable windowed-sinc is applied in axis passes by the callers.
        int axis=e.mode;for(int j=-radius;j<=radius;++j){double w=j?sin(pi*e.a*j)/(pi*j):e.a;w*=.5+.5*cos(pi*j/(radius+1));auto at=(long long)g.coord(i,axis)+j;at=at<0?0:at>=(long long)g.n[axis]?(long long)g.n[axis]-1:at;std::size_t idx=i-g.coord(i,axis)*g.stride[axis]+std::size_t(at)*g.stride[axis];sum+=w*v[idx];weights+=w;}return sum/weights; }
    default:return x;
    }
}
inline void fft_volume(std::vector<Complex>& data,const Grid3& g,bool inverse) {
    for(int a=0;a<3;++a){std::size_t len=g.n[a],stride=g.stride[a],count=g.size/len;auto size=fft_scratch(len);std::vector<Complex> scratch(size);
        for(std::size_t line=0;line<count;++line){auto base=(line/stride)*len*stride+line%stride;fft_line(data.data()+base,len,stride,scratch.data(),size/2,inverse);}}
}
VCR_HD inline bool frequency_selected(const Grid3& g,std::size_t i,double fraction,bool low) {
    bool selected=true;for(int a=0;a<3;++a){auto k=g.coord(i,a);k=k<g.n[a]-k?k:g.n[a]-k;double limit=fraction*double(g.n[a])/2;selected &= double(k)<=limit;}
    return low?selected:!selected;
}
inline std::vector<double> reference_channel(const VolumeIntensity& t,std::size_t channel) {
    auto n=product(Shape(t.reference_shape.begin()+1,t.reference_shape.end()));auto c=t.reference_shape[0]==1?0:channel;
    require(c<t.reference_shape[0],"reference needs one channel or matching image channels");return {t.reference.begin()+c*n,t.reference.begin()+(c+1)*n};
}
} // namespace volume_detail
inline ChannelPlan channel_parameters(const VolumeIntensity& t,const Array& a,Random& r) {
    volume_detail::Grid3 grid(a.spatial());
    double v=draw(t.amount,r),b=draw(t.secondary,r),c=draw(t.scale,r),d=draw(t.extra,r);
    const auto k=t.kind;
    if(k==VolumeKind::BiasField)require(v>=0 && v<=3 && c>0 && c<=1,"bias std must be [0,3], scale (0,1]");
    if(k==VolumeKind::PlasmaBrightnessContrast || k==VolumeKind::PlasmaShadow)require(c>0 && c<=1 && d>0 && d<=1,"fractal scale and persistence must be (0,1]");
    if(k==VolumeKind::PlasmaBrightnessContrast)require(fabs(b)<=1,"plasma contrast must be [-1,1]");
    if(k==VolumeKind::PlasmaShadow || k==VolumeKind::Vignetting || k==VolumeKind::SaltAndPepper || k==VolumeKind::HistogramMatching || k==VolumeKind::PixelDistributionAdaptation || k==VolumeKind::FDA)probability(v);
    if(k==VolumeKind::Illumination)require(fabs(v)<=1 && c>0,"illumination strength must be [-1,1], radius positive");
    if(k==VolumeKind::Vignetting)require(c>0,"vignetting radius must be positive");
    if(k==VolumeKind::SaltAndPepper)probability(b);
    if(k==VolumeKind::MultiplicativeNoise)require(v>=0 && b>=v,"multipliers need 0 <= low <= high");
    if(k==VolumeKind::ShotNoise)require(v>0 && v<=1e9,"photons must be (0,1e9]");
    if(k==VolumeKind::AdditiveNoise)require(v>=0 && c>0 && d>0,"noise scale must be nonnegative, beta shapes positive");
    if(k==VolumeKind::MotionBlur || k==VolumeKind::Defocus || k==VolumeKind::ZoomBlur || k==VolumeKind::UnsharpMask)require(v>=0 && v<=64,"filter strength/radius must be [0,64]");
    if(k==VolumeKind::UnsharpMask)require(b>0 && b<=64,"unsharp sigma must be (0,64]");
    if(k==VolumeKind::GibbsRinging || k==VolumeKind::RingingOvershoot)require(v>0 && v<=1,"frequency cutoff must be (0,1]");
    if(k==VolumeKind::RingingOvershoot)require(b>=1 && b<=64 && floor(b)==b,"sinc radius must be integer [1,64]");
    if(k==VolumeKind::FDA)probability(b);
    if(k==VolumeKind::CLAHE)require(v>=0,"clip limit must be nonnegative");
    if(k==VolumeKind::RandomToneCurve){require(v>=0 && v<=1,"tone curve scale must be [0,1]");b=volume_detail::clamp(2./3+v*r.normal(),0,1);v=volume_detail::clamp(1./3+v*r.normal(),0,1);if(v>b)std::swap(v,b);}
    Point direction{};double norm=0;for(int j=0;j<3;++j){direction[j]=t.direction.empty()?r.normal():t.direction[j];norm+=direction[j]*direction[j];}require(norm>0,"direction must be nonzero");
    ChannelPlan p;p.values={v,b,c,d};for(auto x:direction)p.values.push_back(x/sqrt(norm));
    for(int j=0;j<3;++j)p.values.push_back(t.center.empty()?(t.kind==VolumeKind::Illumination && t.mode==1?double(r.index(2)):.5):t.center[j]);
    p.seed=r.next();return p;
}
inline Parameters sample(const VolumeIntensity& t,const Sample& s,Random& r) {
    auto p=sample_channels(s,t.options,r,[&](const Array& a,Random& rng){return channel_parameters(t,a,rng);});
    if(t.shared_noise){auto seed=r.next();for(auto& c:p.channels)c.seed=split_seed(seed,c.key);}
    return p;
}
inline void apply(const VolumeIntensity& t,Sample& sample,const Parameters& p) {
    using namespace volume_detail;
    for(const auto& channel:p.channels){auto& a=sample.at(channel.key).array;Grid3 g(a.spatial());Effect e(t,channel,g);auto v=a.channel(channel.channel);auto s=stats(v.data(),v.size());Stats rs{};std::vector<double> table;
        if(e.kind==VolumeKind::ShotNoise)require(s.low>=0 && s.high*e.a<=1e12,"shot noise requires nonnegative intensities and rate <= 1e12");
        if(histogram_kind(e.kind)){table.resize((e.kind==VolumeKind::CLAHE?e.tile_count:2)*e.bins);for(std::size_t tile=0;tile<(e.kind==VolumeKind::CLAHE?e.tile_count:1);++tile)prepare_histogram(e,g,v.data(),s,table.data(),tile);}
        if(e.kind==VolumeKind::HistogramMatching || e.kind==VolumeKind::PixelDistributionAdaptation){auto ref=reference_channel(t,channel.channel);rs=stats(ref.data(),ref.size());if(e.kind==VolumeKind::HistogramMatching){histogram(ref.data(),ref.size(),table.data()+e.bins,e.bins,rs.low,rs.high);cdf(table.data()+e.bins,e.bins);}}
        if(e.kind==VolumeKind::UnsharpMask){table=v;gaussian_blur(table,Grid(a.spatial()),std::vector<double>(3,e.b),4,Boundary::Nearest);}
        if(frequency_kind(e.kind)){
            if(e.kind==VolumeKind::GibbsRinging && e.a==1)continue;
            std::vector<Complex> f(g.size);for(std::size_t i=0;i<g.size;++i)f[i].re=v[i];fft_volume(f,g,false);
            if(e.kind==VolumeKind::FDA){require(Shape(t.reference_shape.begin()+1,t.reference_shape.end())==a.spatial(),"FDA reference must match spatial shape");auto ref=reference_channel(t,channel.channel);std::vector<Complex> rf(g.size);for(std::size_t i=0;i<g.size;++i)rf[i].re=ref[i];fft_volume(rf,g,false);
                for(std::size_t i=0;i<g.size;++i)if(frequency_selected(g,i,e.b,true)){double amp=hypot(f[i].re,f[i].im),target=hypot(rf[i].re,rf[i].im);auto unit=amp?f[i].scale(1/amp):Complex{1,0};f[i]=unit.scale((1-e.a)*amp+e.a*target);}}
            else for(std::size_t i=0;i<g.size;++i)if(frequency_selected(g,i,e.a,false))f[i]={};
            fft_volume(f,g,true);for(std::size_t i=0;i<g.size;++i)v[i]=f[i].re;
        }else if(e.kind==VolumeKind::RingingOvershoot){for(int axis=0;axis<3;++axis){e.mode=axis;std::vector<double> out(g.size);for(std::size_t i=0;i<g.size;++i)out[i]=effect_value(e,g,v.data(),i,s,nullptr,rs);v.swap(out);}}
        else {std::vector<double> out(g.size);for(std::size_t i=0;i<g.size;++i)out[i]=effect_value(e,g,v.data(),i,s,table.data(),rs);v.swap(out);}
        for(auto x:v)require(std::isfinite(x),"volumetric augmentation produced nonfinite values");a.channel(channel.channel,v);
    }restore_valid(sample);
}
} // namespace vcr::augment
