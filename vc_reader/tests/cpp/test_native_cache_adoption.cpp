#include "vc_reader/native/disk_cache.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace fs=std::filesystem;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void write(const fs::path& p,const std::string& bytes){fs::create_directories(p.parent_path());std::ofstream out(p);out<<bytes;}
int main() try {
    const auto root=fs::temp_directory_path()/("vcr-adopt-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {fs::path p;~Cleanup(){fs::remove_all(p);}} cleanup{root};
    const auto level=root/"volume/0";
    write(level/".zarray","{}");write(level/".zattrs","{}");
    write(level/"0.0.0",std::string(32,'a'));
    write(level/"0.0.1",std::string(32,'b'));
    fs::last_write_time(level/"0.0.0",fs::file_time_type::clock::now()-std::chrono::hours(2));
    fs::last_write_time(level/"0.0.1",fs::file_time_type::clock::now()-std::chrono::hours(1));
    write(level/"0.0.2.empty","");write(root/"unrelated","untouched");
    write(root/"outside/0.0.0",std::string(100,'c'));
    fs::create_directory_symlink(root/"outside",level/"linked");
    std::vector<std::uint8_t> data(32,7);
    {
        vcr::DiskCache cache(root.string(),64);
        cache.adopt_chunks(root.string());
        cache.adopt_chunks(root.string());
        cache.write((level/"0.0.2").string(),data.data(),data.size());
        check(!fs::exists(level/"0.0.0") && fs::exists(level/"0.0.1"),"adopted FIFO eviction");
    }
    {
        vcr::DiskCache cache(root.string(),32);
        check(!fs::exists(level/"0.0.1") && fs::exists(level/"0.0.2"),"adopted accounting survives reopening");
        cache.set_max_bytes(0);
        check(!fs::exists(level/"0.0.2"),"reconfigured budget evicts adopted files");
    }
    check(fs::exists(level/".zarray") && fs::exists(level/".zattrs") && fs::exists(level/"0.0.2.empty"),"metadata and markers preserved");
    check(fs::exists(root/"unrelated") && fs::exists(root/"outside/0.0.0"),"unrelated and symlink target preserved");
    std::cout<<"cache adoption passed\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
