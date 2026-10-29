#include "vc_reader/slice.hpp"
#include "vc_reader/zarr.hpp"
#include <chrono>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
using namespace vcr;
namespace s = vcr::slice;
void check(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
void compare(const std::vector<float>& a,const std::vector<float>& b) {
    check(a.size()==b.size(),"output size");
    for (std::size_t i=0;i<a.size();++i)
        check((std::isnan(a[i]) && std::isnan(b[i])) || std::abs(a[i]-b[i])<1e-3f,"sample agreement");
}
#ifdef VC_READER_CUDA
template<class T> std::vector<T> download(const device::Storage& storage,std::size_t n) {
    std::vector<T> out(n);
    device::check(cudaMemcpy(out.data(),storage->ptr,n*sizeof(T),cudaMemcpyDeviceToHost));
    return out;
}
#endif
int main() try {
    const auto root=std::filesystem::temp_directory_path()/
        ("vcr-resident-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path p; ~Cleanup(){std::filesystem::remove_all(p);} } cleanup{root};
    for (const auto dtype : {"|u1","<u2","<f4"}) {
        zarr::CreateOptions options;
        options.shape={8,8,12}; options.chunks={4,4,4}; options.dtype=dtype;
        options.compressor="zstd"; options.fillValue=3;
        auto writer=zarr::Writer::create(root/dtype,options);
        const auto type=writer.meta().type;
        // A field with chunk boundaries in all three axes. Last X chunk stays fill.
        for (int cz=0;cz<2;++cz) for(int cy=0;cy<2;++cy) for(int cx=0;cx<2;++cx) {
            std::vector<std::uint8_t> data(64*type.itemsize);
            for(int z=0;z<4;++z) for(int y=0;y<4;++y) for(int x=0;x<4;++x) {
                const float value=float(1+(cz*4+z)*5+(cy*4+y)*3+cx*4+x);
                const auto i=((z*4+y)*4+x)*type.itemsize;
                if(type.itemsize==1) data[i]=std::uint8_t(value);
                else if(type.itemsize==2) {auto v=std::uint16_t(value);std::memcpy(data.data()+i,&v,2);}
                else std::memcpy(data.data()+i,&value,4);
            }
            writer.writeChunk({cz,cy,cx},data.data(),data.size());
        }
        auto opened=zarr::openArray((root/dtype).string(),{});
        auto array=std::shared_ptr<Array>(std::move(opened.array));
        chunks::Service service({.decodedBytes=1<<20,.remoteWorkers=1,.localWorkers=1});
        auto source=service.addSource(std::vector<std::shared_ptr<Array>>{array});
        s::Resident resident{*array,service,source};
        constexpr int W=16,H=12,N=W*H;
        std::array<float,3> origin{-1.25f,-0.5f,3.25f},u{0.9f,0,0},v{0,0.75f,0};
        std::vector<float> expected(N),actual(N);
        std::vector<std::uint8_t> valid(N);
        s::plane(resident,type,origin,u,v,W,H,s::Order::Linear,actual.data(),valid.data());
        check(service.stats().queued==0 && service.stats().fetched==0,"resident sampling never queues or fetches");
        check(valid[W*3+4]==0 && std::isnan(actual[W*3+4]),"cold footprint unavailable");
        for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<3;++x) service.get({source,0,z,y,x});
        for(const auto order : {s::Order::Nearest,s::Order::Linear,s::Order::Cubic}) {
            s::plane(*array,type,origin,u,v,W,H,order,expected.data());
            s::plane(resident,type,origin,u,v,W,H,order,actual.data(),valid.data());
            compare(actual,expected);
            for(auto b:valid) check(b==1,"warm plane coverage including edges and fill");
#ifdef VC_READER_CUDA
            device::Storage gpuValid;
            auto gpu=s::cuda::plane(resident,type,origin,u,v,W,H,order,&gpuValid);
            compare(actual,download<float>(gpu,N));
            check(valid==download<std::uint8_t>(gpuValid,N),"CUDA plane coverage");
#endif
        }
        const float points[]={3.5f,3.5f,3.5f, 9,2,2, NAN,0,0, -100,0,0};
        std::vector<float> pointOut(4),pointRef(4);
        std::vector<std::uint8_t> pointValid(4);
        s::points(resident,type,points,4,s::Order::Cubic,pointOut.data(),pointValid.data());
        s::points(*array,type,points,4,s::Order::Cubic,pointRef.data());
        compare(pointOut,pointRef);
        check(pointValid==std::vector<std::uint8_t>({1,1,0,1}),"point validity");
        std::vector<float> base(N*3),dirs(N*3,0);
        for(int p=0;p<N;++p) {base[3*p]=origin[0]+(p%W)*u[0];base[3*p+1]=origin[1]+(p/W)*v[1];base[3*p+2]=3.25f;dirs[3*p+2]=1;}
        const std::vector<float> offsets{-2,0,2};
        std::vector<float> band(N*3),bandRef(N*3);
        std::vector<std::uint8_t> bandValid(N*3);
        s::band(resident,type,base.data(),dirs.data(),N,W,offsets,s::Order::Linear,band.data(),bandValid.data());
        s::band(*array,type,base.data(),dirs.data(),N,W,offsets,s::Order::Linear,bandRef.data());
        compare(band,bandRef);
#ifdef VC_READER_CUDA
        {
            device::Storage mask;
            auto gpu=s::cuda::band(resident,type,base.data(),dirs.data(),N,W,offsets,s::Order::Linear,&mask);
            compare(band,download<float>(gpu,N*3));
            check(bandValid==download<std::uint8_t>(mask,N*3),"CUDA band coverage");
        }
#endif
        for(const auto method : {s::Composite::Method::Mean,s::Composite::Method::Max,s::Composite::Method::Median,s::Composite::Method::Alpha}) {
            s::Composite config;config.method=method;
            s::composite(resident,type,base.data(),dirs.data(),N,W,offsets,s::Order::Linear,config,actual.data(),valid.data());
            s::composite(*array,type,base.data(),dirs.data(),N,W,offsets,s::Order::Linear,config,expected.data());
            compare(actual,expected);
#ifdef VC_READER_CUDA
            device::Storage gpuValid;
            auto gpu=s::cuda::composite(resident,type,base.data(),dirs.data(),N,W,offsets,s::Order::Linear,config,&gpuValid);
            compare(actual,download<float>(gpu,N));
            check(valid==download<std::uint8_t>(gpuValid,N),"CUDA composite coverage");
#endif
        }
        // Blocking consumers share the service with the viewer. Start cold for
        // each CPU/CUDA request, then verify a resident repaint needs no fetches.
        s::Cached cached{*array,service,source};
        for (const auto order : {s::Order::Nearest,s::Order::Linear,s::Order::Cubic}) {
            service.invalidate(source);
            s::plane(*array,type,origin,u,v,W,H,order,expected.data());
            s::plane(cached,type,origin,u,v,W,H,order,actual.data());
            compare(actual,expected);
            const auto fetched=service.stats().fetched;
            s::plane(resident,type,origin,u,v,W,H,order,actual.data(),valid.data());
            compare(actual,expected);
            for(auto b:valid) check(b==1,"cached request makes viewer footprints resident");
            s::plane(cached,type,origin,u,v,W,H,order,actual.data());
            check(service.stats().fetched==fetched,"cached warm request reuses shared chunks");
#ifdef VC_READER_CUDA
            service.invalidate(source);
            compare(expected,download<float>(s::cuda::plane(cached,type,origin,u,v,W,H,order),N));
#endif
            service.invalidate(source);
            s::points(*array,type,points,4,order,pointRef.data());
            s::points(cached,type,points,4,order,pointOut.data());
            compare(pointOut,pointRef);
#ifdef VC_READER_CUDA
            service.invalidate(source);
            compare(pointRef,download<float>(s::cuda::points(cached,type,points,4,order),4));
#endif
            service.invalidate(source);
            s::band(*array,type,base.data(),dirs.data(),N,W,offsets,order,bandRef.data());
            s::band(cached,type,base.data(),dirs.data(),N,W,offsets,order,band.data());
            compare(band,bandRef);
#ifdef VC_READER_CUDA
            service.invalidate(source);
            compare(bandRef,download<float>(s::cuda::band(cached,type,base.data(),dirs.data(),N,W,offsets,order),N*3));
#endif
            for (const auto method : {s::Composite::Method::Mean,s::Composite::Method::Max,
                                     s::Composite::Method::Min,s::Composite::Method::Median,
                                     s::Composite::Method::MinAbs,s::Composite::Method::Alpha,
                                     s::Composite::Method::BeerLambert}) {
                s::Composite config;config.method=method;
                service.invalidate(source);
                s::composite(*array,type,base.data(),dirs.data(),N,W,offsets,order,config,expected.data());
                s::composite(cached,type,base.data(),dirs.data(),N,W,offsets,order,config,actual.data());
                compare(actual,expected);
#ifdef VC_READER_CUDA
                service.invalidate(source);
                compare(expected,download<float>(s::cuda::composite(cached,type,base.data(),dirs.data(),N,W,offsets,order,config),N));
#endif
            }
        }
        // Even when chunks are evicted as they are fetched, each wave retains
        // the data it needs until sampling finishes.
        service.configure(1,1,false);
        s::plane(*array,type,origin,u,v,W,H,s::Order::Cubic,expected.data());
        s::plane(cached,type,origin,u,v,W,H,s::Order::Cubic,actual.data());
        compare(actual,expected);
        check(service.stats().decodedBytes<=1 && service.stats().evicted>0,"bounded shared cache");
#ifdef VC_READER_CUDA
        compare(expected,download<float>(s::cuda::plane(cached,type,origin,u,v,W,H,s::Order::Cubic),N));
#endif
        service.configure(1<<20,1,false);
        const auto failed=service.addSource([](std::int32_t,std::int64_t,std::int64_t,std::int64_t)->ChunkPtr {
            throw std::runtime_error("fixture fetch error");
        },1);
        s::Cached broken{*array,service,failed};
        const float inside[]={1,1,1};
        bool threw=false;
        try {s::points(broken,type,inside,1,s::Order::Nearest,actual.data());}
        catch(const std::runtime_error& e) {threw=std::string(e.what()).find("fixture fetch error")!=std::string::npos;}
        check(threw,"blocking sampling propagates fetch errors");
#ifdef VC_READER_CUDA
        threw=false;
        try {(void)s::cuda::points(broken,type,inside,1,s::Order::Nearest);}
        catch(const std::runtime_error& e) {threw=std::string(e.what()).find("fixture fetch error")!=std::string::npos;}
        check(threw,"CUDA blocking sampling propagates fetch errors");
#endif
        service.removeSource(failed);
        service.invalidate(source);
        service.get({source,0,0,0,0});
        const float boundary[]={3.5f,1,1};
        float value;std::uint8_t coverage;
        s::points(resident,type,boundary,1,s::Order::Linear,&value,&coverage);
        check(!coverage && std::isnan(value),"interpolation crosses unavailable chunk");
        const float partialBase[]={1,1,1}, direction[]={1,0,0};
        const std::vector<float> partialOffsets{0,4};
        s::Composite partialConfig;
        s::composite(resident,type,partialBase,direction,1,1,partialOffsets,s::Order::Nearest,partialConfig,&value,&coverage);
        check(!coverage && std::isnan(value),"incomplete composites cannot look complete");
        // The consumer can fill an invalid fine-level pixel from a ready coarse level.
        chunks::Service fallbackService({.decodedBytes=1<<20,.remoteWorkers=1,.localWorkers=1});
        auto pyramid=fallbackService.addSource(std::vector<std::shared_ptr<Array>>{array,array});
        fallbackService.get({pyramid,1,0,0,0});
        s::Resident fine{*array,fallbackService,pyramid,0}, coarse{*array,fallbackService,pyramid,1};
        float fineValue,coarseValue;std::uint8_t fineValid,coarseValid;
        s::points(fine,type,partialBase,1,s::Order::Nearest,&fineValue,&fineValid);
        s::points(coarse,type,partialBase,1,s::Order::Nearest,&coarseValue,&coarseValid);
        check(!fineValid && coarseValid && std::isfinite(coarseValue),"progressive level fallback");
        if(type.kind=='f') {
            std::vector<float> nanChunk(64,NAN);
            writer.writeChunk({0,0,0},reinterpret_cast<const std::uint8_t*>(nanChunk.data()),nanChunk.size()*sizeof(float));
            service.invalidate(source);
            service.get({source,0,0,0,0});
            s::points(resident,type,partialBase,1,s::Order::Nearest,&value,&coverage);
            check(coverage && std::isnan(value),"stored NaN remains available");
        }
#ifdef VC_READER_CUDA
        device::Storage gpuValid;
        auto gpu=s::cuda::points(resident,type,boundary,1,s::Order::Linear,&gpuValid);
        check(download<std::uint8_t>(gpuValid,1)[0]==0 && std::isnan(download<float>(gpu,1)[0]),"CUDA missing footprint");
#endif
    }
    std::cout<<"resident sampling passed\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
