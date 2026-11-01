#include "vc/lasagna/PredictionPooling.hpp"
#include "vc/core/types/Volume.hpp"
#include <opencv2/core.hpp>
#include <Eigen/Eigenvalues>
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <list>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>

#ifdef VC_POOLING_LAPACK
extern "C" void dsyevd_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*,const int*,int*,size_t,size_t);
#endif

namespace vc::lasagna {
namespace {
constexpr std::array<const char*,7> names{"presence","nx","ny","width","sheet_nx","sheet_ny","sheet_grad_mag"};
using Bundle = std::array<std::vector<std::byte>,7>;
using Index = std::array<size_t,3>;
std::atomic<uint64_t> computations{0},hits{0},readNs{0},poolNs{0};
using Clock=std::chrono::steady_clock;
uint64_t elapsed(Clock::time_point start) {return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();}
uint8_t quantize(float x) { return static_cast<uint8_t>(std::clamp(std::nearbyint(x),0.f,255.f)); }
cv::Vec3f axis(uint8_t x, uint8_t y) {
    const float a=(float(x)-128.f)/127.f,b=(float(y)-128.f)/127.f;
    return {a,b,std::sqrt(std::max(0.f,1.f-a*a-b*b))};
}
struct Source {
    std::array<std::unique_ptr<utils::ZarrArray>,7> arrays;
    Index shape{}, chunk{};
    explicit Source(const std::filesystem::path& root,const nlohmann::json& config) {
        for(size_t c=0;c<7;++c) {
            const auto path=config.contains("channels")?
                std::filesystem::path(config.at("channels").at(names[c]).get<std::string>()):
                root/(std::string("fiber_")+names[c]+".ome.zarr")/config.value("source_level",std::string("1"));
            const auto metadata=nlohmann::json::parse(utils::FileSystemStore(path).get_string(".zarray"));
            if(metadata.value("order",std::string("C"))!="C")
                throw std::runtime_error("native pooling requires C-order source");
            arrays[c]=std::make_unique<utils::ZarrArray>(utils::ZarrArray::open(
                path,vc::buildZarrCodecRegistry(c==3?4:1)));
            const auto& m=arrays[c]->metadata();
            if(m.shape.size()!=3 || m.chunks.size()!=3 ||
               m.dtype!=(c==3?utils::ZarrDtype::float32:utils::ZarrDtype::uint8))
                throw std::runtime_error("native pooling requires C-order 3D u8 channels and float32 width");
            if(c==0) for(size_t d=0;d<3;++d) {shape[d]=m.shape[d];chunk[d]=m.chunks[d];}
            for(size_t d=0;d<3;++d)
                if(m.shape[d]!=shape[d] || m.chunks[d]!=chunk[d] || chunk[d]%2)
                    throw std::runtime_error("native pooling requires aligned even source chunks");
        }
    }
    std::shared_ptr<const Bundle> compute(const Index& key) const {
        const auto start=Clock::now();
        Bundle input;
        Index valid{}, out{};
        for(size_t d=0;d<3;++d) {
            if(key[d]*chunk[d]>=shape[d]) throw std::runtime_error("pooled chunk outside volume");
            valid[d]=std::min(chunk[d],shape[d]-key[d]*chunk[d]); out[d]=chunk[d]/2;
        }
        const size_t size=out[0]*out[1]*out[2];
        auto result=std::make_shared<Bundle>();
        for(size_t c=0;c<7;++c) {
            auto data=arrays[c]->read_chunk(key);
            if(!data) throw std::runtime_error("unpublished native pooling source: "+arrays[c]->chunk_path(key).string());
            input[c]=std::move(*data);
            if(input[c].size()!=chunk[0]*chunk[1]*chunk[2]*(c==3?4:1))
                throw std::runtime_error("invalid source chunk byte count");
            (*result)[c].resize(size*(c==3?4:1));
            // Match source fill for padding beyond the volume, not invented evidence.
            if(c!=3) std::fill((*result)[c].begin(),(*result)[c].end(),
                              static_cast<std::byte>(arrays[c]->metadata().fill_value.value_or(0)));
            else {
                const float fill=float(arrays[c]->metadata().fill_value.value_or(0));
                for(size_t i=0;i<size;++i) std::memcpy((*result)[c].data()+i*4,&fill,4);
            }
        }
        readNs+=elapsed(start);
        const auto poolingStart=Clock::now();
        auto byte=[&](size_t c,size_t i){return std::to_integer<uint8_t>(input[c][i]);};
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
            const size_t k=std::min(size_t(2),n), dst=(z*out[1]+y)*out[2]+x;
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
            // Keep ZYX float accumulation order, including unselected zero terms.
            for(size_t c: {size_t(3),size_t(6)}) {
                float sum=0;
                for(size_t j=0;j<n;++j) {
                    float value;
                    if(c==3) std::memcpy(&value,input[c].data()+indices[j]*4,4); else value=byte(c,indices[j]);
                    sum+=value*weights[j];
                }
                const float value=(sum/float(n))/std::max(denom/float(n),1e-12f);
                if(c==3) std::memcpy((*result)[c].data()+dst*4,&value,4); else (*result)[c][dst]=std::byte(quantize(value));
            }
            float presence=0; for(size_t j=0;j<k;++j) presence+=byte(0,indices[rank[j]]);
            (*result)[0][dst]=std::byte(quantize(presence/float(k)));
            for(size_t c: {size_t(1),size_t(4)}) {
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
                (*result)[c][dst]=std::byte(quantize(v[0]*127.f+128.f));
                (*result)[c+1][dst]=std::byte(quantize(v[1]*127.f+128.f));
            }
        }
        poolNs+=elapsed(poolingStart);
        const auto count=++computations;
        if(count%1024==0) std::clog<<"native-pool chunks="<<count<<" hits="<<hits.load()
            <<" read_task_wall_s="<<double(readNs.load())/1e9<<" pool_task_wall_s="<<double(poolNs.load())/1e9<<std::endl;
        return result;
    }
};

// One process-wide bounded bundle LRU, shared across source and channel views.
// Native ChunkCache owns the ordinary decoded per-channel cache separately.
struct Cache {
    struct Entry { std::shared_future<std::shared_ptr<const Bundle>> future; size_t bytes=0; std::list<std::string>::iterator lru; };
    std::mutex mutex;
    std::unordered_map<std::string,Entry> entries;
    std::list<std::string> lru;
    size_t bytes=0;
    static constexpr size_t capacity=size_t(8)<<30;
    std::shared_ptr<const Bundle> get(const std::string& id,const Source& source,const Index& key) {
        std::shared_future<std::shared_ptr<const Bundle>> future;
        std::shared_ptr<std::promise<std::shared_ptr<const Bundle>>> promise;
        {
            std::lock_guard lock(mutex);
            if(auto it=entries.find(id);it!=entries.end()) {
                ++hits;
                future=it->second.future;
                if(it->second.bytes) lru.splice(lru.begin(),lru,it->second.lru);
            } else {
                promise=std::make_shared<std::promise<std::shared_ptr<const Bundle>>>();
                future=promise->get_future().share();entries.emplace(id,Entry{future,0,{}});
            }
        }
        if(!promise) return future.get(); // no cache lock, owner computes inline
        try {
            auto bundle=source.compute(key);size_t count=0;for(const auto& v:*bundle)count+=v.size();
            {
                std::lock_guard lock(mutex);
                lru.push_front(id);auto& entry=entries.at(id);entry.bytes=count;entry.lru=lru.begin();bytes+=count;
                while(bytes>capacity && lru.size()>1) {
                    const auto last=lru.back();bytes-=entries.at(last).bytes;entries.erase(last);lru.pop_back();
                }
            }
            promise->set_value(bundle);return bundle;
        } catch(...) {
            promise->set_exception(std::current_exception());
            std::lock_guard lock(mutex);entries.erase(id);throw;
        }
    }
};
Cache cache;
std::mutex sourcesMutex;
std::unordered_map<std::string,std::weak_ptr<Source>> sources;
class PooledStore final:public utils::Store {
    utils::FileSystemStore metadata;
    std::shared_ptr<Source> source;
    std::string identity;
    size_t channel=0;
public:
    explicit PooledStore(const std::filesystem::path& path):metadata(path) {
        const auto config=nlohmann::json::parse(metadata.get_string(".native_pool.json"));
        if(config.at("revision")!=1 || config.at("factor")!=2 || config.at("mode")!="top2_spatial_comp")
            throw std::runtime_error("unsupported native prediction pooling recipe");
        const std::string name=config.at("channel");
        auto it=std::find(names.begin(),names.end(),name);
        if(it==names.end()) throw std::runtime_error("unknown native pooling channel");channel=it-names.begin();
        const auto root=std::filesystem::canonical(config.at("source").get<std::string>());
        auto sourceConfig=config;
        if(sourceConfig.contains("channels"))
            for(auto& [key,value]:sourceConfig["channels"].items()) {
                auto p=std::filesystem::path(value.get<std::string>());
                value=std::filesystem::canonical(p.is_absolute()?p:root/p).string();
            }
        identity=root.string()+"/"+sourceConfig.value("source_level",std::string("1"))+"/"+
            sourceConfig.value("channels",nlohmann::json::object()).dump();
        {
            std::lock_guard lock(sourcesMutex);
            source=sources[identity].lock();
        }
        if(!source) {
            auto loaded=std::make_shared<Source>(root,sourceConfig);
            std::lock_guard lock(sourcesMutex);
            source=sources[identity].lock();
            if(!source) {source=std::move(loaded);sources[identity]=source;}
        }
        const auto meta=nlohmann::json::parse(metadata.get_string(".zarray"));
        if(meta.at("dtype")!=(channel==3?"<f4":"|u1"))
            throw std::runtime_error("native pooled view dtype mismatch");
        if(!meta.at("compressor").is_null() || !meta.at("filters").is_null() ||
           meta.at("order")!="C" || meta.at("dimension_separator")!="/")
            throw std::runtime_error("native pooled view requires raw C-order metadata");
        for(size_t d=0;d<3;++d)
            if(meta.at("shape")[d]!=(source->shape[d]+1)/2 || meta.at("chunks")[d]!=source->chunk[d]/2)
                throw std::runtime_error("native pooled view geometry mismatch");
        identity+="/top2_spatial_comp/v1/";
    }
    bool exists(const std::string& key)const override {return get_if_exists(key).has_value();}
    std::optional<std::vector<std::byte>> get_if_exists(const std::string& key)const override {
        if(key.empty() || key[0]=='.' || key=="zarr.json") return metadata.get_if_exists(key);
        Index index{};std::istringstream stream(key);char a,b;
        if(!(stream>>index[0]>>a>>index[1]>>b>>index[2]) || a!='/' || b!='/' || stream.peek()!=EOF)
            throw std::runtime_error("invalid native pooled chunk key: "+key);
        const auto bundle=cache.get(identity+key,*source,index);
        return (*bundle)[channel];
    }
    std::vector<std::byte> get(const std::string& key)const override {
        auto result=get_if_exists(key);if(!result)throw std::runtime_error("missing pooled metadata: "+key);return std::move(*result);
    }
    void set(const std::string&,std::span<const std::byte>)override {throw std::runtime_error("read-only pooled view");}
    void erase(const std::string&)override {throw std::runtime_error("read-only pooled view");}
};
}
std::shared_ptr<utils::Store> openPooledPredictionStore(const std::filesystem::path& path) {
    return std::make_shared<PooledStore>(path);
}
}
