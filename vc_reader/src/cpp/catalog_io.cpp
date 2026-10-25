#include "catalog_io.hpp"
#include "vc_reader/catalog.hpp"
#include "http.hpp"
#include <zlib.h>
#include <fstream>
#include <atomic>
#include <chrono>
#include <mutex>
#include <sstream>
#include <thread>
namespace vcr::catalog {
namespace {
constexpr size_t maxBytes=512*1024*1024;
std::string decode(std::string body) {
    if(body.size()<2 || (unsigned char)body[0]!=0x1f || (unsigned char)body[1]!=0x8b)return body;
    z_stream z{};
    if(inflateInit2(&z,16+MAX_WBITS)!=Z_OK)throw std::runtime_error("cannot initialize catalog decompression");
    struct End { z_stream* z; ~End(){inflateEnd(z);} } end{&z};
    z.next_in=reinterpret_cast<Bytef*>(body.data());z.avail_in=body.size();
    std::string output;std::array<char,65536> block;
    int status;
    do {
        z.next_out=reinterpret_cast<Bytef*>(block.data());z.avail_out=block.size();
        status=inflate(&z,Z_NO_FLUSH);
        if(status!=Z_OK && status!=Z_STREAM_END)throw std::runtime_error("invalid gzip catalog");
        output.append(block.data(),block.size()-z.avail_out);
        if(output.size()>maxBytes)throw std::runtime_error("catalog exceeds size limit");
    }while(status!=Z_STREAM_END);
    return output;
}
void writeAtomic(const std::filesystem::path& path, const std::string& value) {
    if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
    static std::atomic<uint64_t> sequence{0};
    const auto temp=path.string()+".tmp-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(sequence++);
    try {
        {std::ofstream out(temp,std::ios::binary);out.exceptions(std::ios::badbit|std::ios::failbit);out.write(value.data(),value.size());out.close();}
        std::filesystem::rename(temp,path);
    }catch(...){std::error_code ec;std::filesystem::remove(temp,ec);throw;}
}
std::pair<std::string,HttpResponse> request(const std::string& url,const std::string& etag={},double timeout=60) {
    if(!std::isfinite(timeout)||timeout<=0||timeout>86400)throw std::invalid_argument("catalog timeout must be in (0, 86400] seconds");
    HttpOptions options;options.request_timeout_ms=std::max(1L,long(timeout*1000));options.retries=0;
    HttpClient client(options);std::string body;
    std::vector<std::pair<std::string,std::string>> headers;
    if(!etag.empty())headers.emplace_back("If-None-Match",etag);
    const auto response=client.get_stream(url,[&]{body.clear();},[&](const char* p,size_t n){
        if(n>maxBytes-body.size())throw std::runtime_error("catalog exceeds size limit");body.append(p,n);
    },headers);
    if(response.status==404)throw std::runtime_error("catalog not found: "+url);
    return {decode(std::move(body)),response};
}
}
std::string readCatalogFile(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot open catalog: "+path.string());
    std::string body;std::array<char,65536> block;
    while(in){in.read(block.data(),block.size());body.append(block.data(),in.gcount());if(body.size()>maxBytes)throw std::runtime_error("catalog exceeds size limit");}
    if(!in.eof())throw std::runtime_error("cannot read catalog: "+path.string());
    return decode(std::move(body));
}
std::string fetchCatalogText(const std::string& url) {return request(url).first;}
CatalogLoadResult loadCatalog(const std::string& source,const CatalogLoadOptions& options) {
    if(!source.starts_with("https://")&&!source.starts_with("http://"))return {loadOpenDataManifestFile(source),false,{}};
    static std::mutex cacheMutex;
    std::lock_guard lock(cacheMutex);
    std::optional<OpenDataManifest> cached;
    nlohmann::json info=nlohmann::json::object();
    const auto now=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto infoPath=options.cacheFile.string()+".info.json";
    if(!options.cacheFile.empty())try {
        info=nlohmann::json::parse(readCatalogFile(infoPath));
        if(info.value("source",std::string{})==source)cached=loadOpenDataManifestFile(options.cacheFile,source);
    }catch(const std::exception&){}
    if(cached&&!options.refresh&&now-info.value("checked_at",0.)<options.maxAgeSeconds)return {*cached,true,{}};
    try {
        auto [body,response]=request(source,cached?info.value("etag",std::string{}):std::string{},options.timeoutSeconds);
        if(response.status==304&&!cached)throw std::runtime_error("catalog returned 304 without cached content");
        auto manifest=response.status==304?*cached:parseOpenDataManifest(body,source);
        if(!options.cacheFile.empty()) {
            if(response.status!=304)writeAtomic(options.cacheFile,body);
            writeAtomic(infoPath,nlohmann::json{{"source",source},{"checked_at",now},{"etag",response.etag.empty()?info.value("etag",std::string{}):response.etag}}.dump());
        }
        return {std::move(manifest),response.status==304,{}};
    }catch(const std::exception& e){if(cached)return {*cached,true,e.what()};throw;}
}
}
