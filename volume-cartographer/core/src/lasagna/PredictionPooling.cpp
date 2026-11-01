#include "vc/lasagna/PredictionPooling.hpp"
#include <vc_reader/computed.hpp>
#include <vc_reader/zarr_write.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Eigenvalues>
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "vc_reader/zarr.hpp"

#ifdef VC_POOLING_LAPACK
extern "C" void dsyevd_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*,const int*,int*,size_t,size_t);
#endif

namespace vc::lasagna {
namespace {
constexpr std::array<const char*,7> names{"presence","nx","ny","width","sheet_nx","sheet_ny","sheet_grad_mag"};
using Index = std::array<size_t,3>;
std::atomic<uint64_t> computations{0},readNs{0},poolNs{0};
using Clock=std::chrono::steady_clock;
uint64_t elapsed(Clock::time_point start) {return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();}
uint8_t quantize(float x) { return static_cast<uint8_t>(std::clamp(std::nearbyint(x),0.f,255.f)); }
cv::Vec3f axis(uint8_t x, uint8_t y) {
    const float a=(float(x)-128.f)/127.f,b=(float(y)-128.f)/127.f;
    return {a,b,std::sqrt(std::max(0.f,1.f-a*a-b*b))};
}
// The source channels a pooled channel reads: presence and its axis rank and
// weight every voxel; nx/ny and sheet nx/ny pool as axis pairs.
std::vector<size_t> inputsOf(size_t channel) {
    switch(channel) {
    case 0: case 1: case 2: return {0,1,2};
    case 4: case 5: return {0,1,2,4,5};
    default: return {0,1,2,channel};
    }
}
struct Source {
    std::array<vcr::zarr::OpenedArray,7> arrays;
    Index shape{}, chunk{};
    Source(const std::filesystem::path& root,const nlohmann::json& config,std::shared_ptr<vcr::MemoryCache> cache) {
        vcr::zarr::OpenOptions options;
        options.memoryCache=std::move(cache);
        for(size_t c=0;c<7;++c) {
            const auto path=config.contains("channels")?
                std::filesystem::path(config.at("channels").at(names[c]).get<std::string>()):
                root/(std::string("fiber_")+names[c]+".ome.zarr")/config.value("source_level",std::string("1"));
            arrays[c]=vcr::zarr::openArray(path.string(),options);
            const auto& m=arrays[c].meta;
            if(m.order!="C")
                throw std::runtime_error("native pooling requires C-order source");
            if(m.shape.size()!=3 || m.chunks.size()!=3 || m.dtype!=(c==3?"<f4":"|u1"))
                throw std::runtime_error("native pooling requires C-order 3D u8 channels and float32 width");
            if(c==0) for(size_t d=0;d<3;++d) {shape[d]=size_t(m.shape[d]);chunk[d]=size_t(m.chunks[d]);}
            for(size_t d=0;d<3;++d)
                if(size_t(m.shape[d])!=shape[d] || size_t(m.chunks[d])!=chunk[d] || chunk[d]%2)
                    throw std::runtime_error("native pooling requires aligned even source chunks");
        }
    }
    // Pools one output chunk of `channel` (chunk/2 voxels per axis) from source chunk `key`.
    void compute(const Index& key, size_t channel, uint8_t* out) const {
        const auto start=Clock::now();
        std::array<std::vector<uint8_t>,7> input;
        Index valid{}, half{};
        for(size_t d=0;d<3;++d) {
            if(key[d]*chunk[d]>=shape[d]) throw std::runtime_error("pooled chunk outside volume");
            valid[d]=std::min(chunk[d],shape[d]-key[d]*chunk[d]); half[d]=chunk[d]/2;
        }
        const size_t size=half[0]*half[1]*half[2];
        const vcr::Index index{std::int64_t(key[0]),std::int64_t(key[1]),std::int64_t(key[2])};
        for(const size_t c: inputsOf(channel)) {
            input[c].resize(arrays[c].array->chunk_bytes());
            if(!arrays[c].array->read_chunk(index,input[c].data(),input[c].size()))
                throw std::runtime_error("unpublished native pooling source: "+arrays[c].location+" chunk "+
                    std::to_string(key[0])+"/"+std::to_string(key[1])+"/"+std::to_string(key[2]));
        }
        // Match source fill for padding beyond the volume, not invented evidence.
        const auto& fill=arrays[channel].meta.fill;
        for(size_t i=0;i<size;++i) std::memcpy(out+i*fill.size(),fill.data(),fill.size());
        readNs+=elapsed(start);
        const auto poolingStart=Clock::now();
        auto byte=[&](size_t c,size_t i){return input[c][i];};
        for(size_t z=0;z<(valid[0]+1)/2;++z) for(size_t y=0;y<(valid[1]+1)/2;++y) for(size_t x=0;x<(valid[2]+1)/2;++x) {
            std::array<size_t,8> indices{}; std::array<cv::Vec3f,8> deltas{};
            size_t n=0;
            for(size_t dz=0;dz<2;++dz) for(size_t dy=0;dy<2;++dy) for(size_t dx=0;dx<2;++dx) {
                if(2*z+dz>=valid[0] || 2*y+dy>=valid[1] || 2*x+dx>=valid[2]) continue;
                indices[n]=((2*z+dz)*chunk[1]+2*y+dy)*chunk[2]+2*x+dx;
                deltas[n++]={float(dx)-.5f,float(dy)-.5f,float(dz)-.5f};
            }
            std::array<size_t,8> rank{}; std::iota(rank.begin(),rank.end(),0);
            std::stable_sort(rank.begin(),rank.begin()+n,[&](size_t a,size_t b){return byte(0,indices[a])>byte(0,indices[b]);});
            const size_t k=std::min(size_t(2),n), dst=(z*half[1]+y)*half[2]+x;
            auto ref=axis(byte(1,indices[rank[0]]),byte(2,indices[rank[0]]));
            ref*=1.f/std::max(std::sqrt(ref.dot(ref)),1e-12f);
            std::array<float,8> weights{};
            for(size_t j=0;j<k;++j) {
                const size_t r=rank[j]; const float along=ref.dot(deltas[r]);
                const float g=std::exp(-.5f*(along*along+std::max(0.f,deltas[r].dot(deltas[r])-along*along)/.25f));
                weights[r]=float(byte(0,indices[r]))*g;
            }
            float denom=0; for(size_t j=0;j<n;++j) denom+=weights[j];
            if(denom==0) {for(size_t j=0;j<k;++j) weights[rank[j]]=1.f;denom=float(k);}
            if(channel==3 || channel==6) {
                // Keep ZYX float accumulation order, including unselected zero terms.
                float sum=0;
                for(size_t j=0;j<n;++j) {
                    float value;
                    if(channel==3) std::memcpy(&value,input[channel].data()+indices[j]*4,4); else value=byte(channel,indices[j]);
                    sum+=value*weights[j];
                }
                const float value=(sum/float(n))/std::max(denom/float(n),1e-12f);
                if(channel==3) std::memcpy(out+dst*4,&value,4); else out[dst]=quantize(value);
            } else if(channel==0) {
                float presence=0; for(size_t j=0;j<k;++j) presence+=byte(0,indices[rank[j]]);
                out[dst]=quantize(presence/float(k));
            } else {
                const size_t c=channel==1 || channel==2 ? 1 : 4;
                cv::Matx33f m=cv::Matx33f::zeros();
                for(size_t j=0;j<n;++j) {
                    auto v=axis(byte(c,indices[j]),byte(c+1,indices[j]));
                    for(int a=0;a<3;++a) for(int b=0;b<3;++b) m(a,b)+=(v[a]*v[b])*weights[j];
                }
                m*=1.f/float(n);
                // Fixed-size double precision with no BLAS calls or shared
                // allocation locks. DSYEVD's OpenBLAS allocator contends here
                // when many independent chunks pool their direction tensors.
                Eigen::Matrix3d matrix;
                for(int a=0;a<3;++a)for(int b=0;b<3;++b)matrix(a,b)=m(a,b);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(matrix);
                if(solver.info()!=Eigen::Success) throw std::runtime_error("native pooling eigensolver failed");
                int winner=0;for(int a=1;a<3;++a)if(float(solver.eigenvalues()[a])>float(solver.eigenvalues()[winner]))winner=a;
                const auto vector=solver.eigenvectors().col(winner);
                cv::Vec3f v{float(vector[0]),float(vector[1]),float(vector[2])};
                // Hemisphere encoding is discontinuous at z=0, and an exactly
                // degenerate leading eigenspace has no unique axis. Retain the
                // reference solver's convention in these rare cases. This
                // threshold selects the solver, never changes an acceptance
                // threshold or snaps/approximates the resulting vector.
                #ifdef VC_POOLING_LAPACK
                if(std::abs(v[2])<1e-6f ||
                   solver.eigenvalues()[2]-solver.eigenvalues()[1] <=
                       1e-6*std::max(1.,std::abs(solver.eigenvalues()[2]))) {
                    double a[9],values[3],work[64];int iw[32],info=0;
                    const int dim=3,lwork=64,liwork=32;const char job='V',uplo='L';
                    for(int row=0;row<3;++row)for(int col=0;col<3;++col)a[row+3*col]=m(row,col);
                    dsyevd_(&job,&uplo,&dim,a,&dim,values,work,&lwork,iw,&liwork,&info,1,1);
                    if(info) throw std::runtime_error("native pooling reference eigensolver failed");
                    winner=0;for(int i=1;i<3;++i)if(float(values[i])>float(values[winner]))winner=i;
                    v={float(a[3*winner]),float(a[3*winner+1]),float(a[3*winner+2])};
                }
                #endif
                if(v[2]<0) v=-v;
                v*=1.f/std::sqrt(std::max(v.dot(v),1e-12f));
                out[dst]=quantize(v[channel==c?0:1]*127.f+128.f);
            }
        }
        poolNs+=elapsed(poolingStart);
        const auto count=++computations;
        if(count%1024==0) std::clog<<"native-pool chunks="<<count
            <<" read_task_wall_s="<<double(readNs.load())/1e9<<" pool_task_wall_s="<<double(poolNs.load())/1e9<<std::endl;
    }
};

// Channel views of one source share its opened arrays.
std::mutex sourcesMutex;
std::unordered_map<std::string,std::weak_ptr<Source>> sources;

nlohmann::json readJson(const std::filesystem::path& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot read "+path.string());
    return nlohmann::json::parse(in);
}
}

vcr::zarr::OpenedArray openPooledPredictionArray(
    const std::filesystem::path& path, std::shared_ptr<vcr::MemoryCache> cache) {
    const auto config=readJson(path/".native_pool.json");
    if(config.at("revision")!=1 || config.at("factor")!=2 || config.at("mode")!="top2_spatial_comp")
        throw std::runtime_error("unsupported native prediction pooling recipe");
    const std::string name=config.at("channel");
    auto it=std::find(names.begin(),names.end(),name);
    if(it==names.end()) throw std::runtime_error("unknown native pooling channel");
    const size_t channel=size_t(it-names.begin());
    const auto root=std::filesystem::canonical(config.at("source").get<std::string>());
    auto sourceConfig=config;
    if(sourceConfig.contains("channels"))
        for(auto& [key,value]:sourceConfig["channels"].items()) {
            auto p=std::filesystem::path(value.get<std::string>());
            value=std::filesystem::canonical(p.is_absolute()?p:root/p).string();
        }
    const std::string identity=root.string()+"/"+sourceConfig.value("source_level",std::string("1"))+"/"+
        sourceConfig.value("channels",nlohmann::json::object()).dump();
    std::shared_ptr<Source> source;
    {
        std::lock_guard lock(sourcesMutex);
        source=sources[identity].lock();
        if(!source) {source=std::make_shared<Source>(root,sourceConfig,cache);sources[identity]=source;}
    }
    vcr::zarr::OpenedArray opened;
    opened.location=path.string();
    const auto zarray=vcr::zarr::Store(path).read(".zarray");
    if(!zarray) throw std::runtime_error("no .zarray at "+path.string());
    opened.meta=vcr::zarr::parseArrayMeta(nlohmann::json::parse(*zarray));
    const auto& meta=opened.meta;
    if(meta.dtype!=(channel==3?"<f4":"|u1"))
        throw std::runtime_error("native pooled view dtype mismatch");
    if(!meta.compressor.is_null() || !meta.filters.is_null() || meta.order!="C" || meta.separator!="/")
        throw std::runtime_error("native pooled view requires raw C-order metadata");
    for(size_t d=0;d<3;++d)
        if(meta.shape[d]!=std::int64_t((source->shape[d]+1)/2) || meta.chunks[d]!=std::int64_t(source->chunk[d]/2))
            throw std::runtime_error("native pooled view geometry mismatch");
    vcr::computed::Options options;
    options.shape=opened.meta.shape;
    options.chunks=opened.meta.chunks;
    options.dtype=opened.meta.dtype;
    options.fillValue=meta.raw.at("fill_value");
    // The memory-cache namespace of the view's chunks.
    options.identity=identity+"/top2_spatial_comp/v1/"+name;
    options.memoryCache=std::move(cache);
    opened.array=vcr::computed::open(options,[source,channel](const vcr::Index& chunk,std::uint8_t* out) {
        source->compute({size_t(chunk[0]),size_t(chunk[1]),size_t(chunk[2])},channel,out);
        return true;
    });
    return opened;
}
}
