#include "vc_reader/native/disk_cache.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace fs=std::filesystem;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void write(const fs::path& p,const std::string& bytes){fs::create_directories(p.parent_path());std::ofstream out(p);out<<bytes;}
// No byte limit: nothing is walked until a limit or a total needs the index.
void unbounded(const fs::path& root)
{
    const auto level=root/"volume/0";
    write(level/".zarray","{}");
    write(level/"0.0.0",std::string(32,'a'));
    write(level/"0.0.1",std::string(32,'b'));
    constexpr auto none=std::numeric_limits<std::int64_t>::max();
    {
        vcr::DiskCache cache(root.string(),none);
        cache.adopt_chunks((root/"volume").string());
        check(cache.total_bytes()==64,"a total walks the files when asked");
    }
    {
        vcr::DiskCache cache(root.string(),none);
        check(fs::exists(level/"0.0.0"),"unbounded opens evict nothing");
        cache.set_max_bytes(32);
        check(cache.total_bytes()==32,"a new limit walks the adopted files and applies");
        check(!fs::exists(level/"0.0.0")||!fs::exists(level/"0.0.1"),"oldest chunk evicted");
    }
    vcr::DiskCache again(root.string(),-1);
    check(again.max_bytes()==32&&again.total_bytes()==32,"a kept limit reconciles on open");
}

// A root nobody limited is unlimited, and whatever is set stays until changed.
void defaultBudget(const fs::path& root)
{
    constexpr auto none=std::numeric_limits<std::int64_t>::max();
    {
        vcr::DiskCache cache(root.string(),-1);
        check(cache.max_bytes()==none,"a new budget is unlimited");
        cache.set_max_bytes(2048);
    }
    {
        vcr::DiskCache cache(root.string(),-1);
        check(cache.max_bytes()==2048,"a limit the user set is kept");
        cache.set_max_bytes(none);
    }
    vcr::DiskCache cache(root.string(),-1);
    check(cache.max_bytes()==none,"the user can lift a limit");
}

void setStored(const fs::path& root,std::int64_t limit)
{
    sqlite3* db=nullptr;
    check(sqlite3_open((root/".vc_reader_cache.sqlite3").c_str(),&db)==SQLITE_OK,"open the index");
    const auto sql="UPDATE budget SET max_bytes="+std::to_string(limit)+" WHERE id=1; PRAGMA user_version=0;";
    const int rc=sqlite3_exec(db,sql.c_str(),nullptr,nullptr,nullptr);
    sqlite3_close(db);
    check(rc==SQLITE_OK,"edit the index");
}

// A root indexed before budgets defaulted to unlimited holds the old 1 GiB seed, which
// nobody chose. It must not evict; any other stored limit was chosen and stays.
void seededByOlderVersion(const fs::path& root)
{
    constexpr auto none=std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t seed=1073741824;
    { vcr::DiskCache cache(root.string(),-1); }
    setStored(root,seed);
    {
        vcr::DiskCache cache(root.string(),-1);
        check(cache.max_bytes()==none,"the old seed becomes unlimited");
        cache.set_max_bytes(seed);
    }
    {
        vcr::DiskCache cache(root.string(),-1);
        check(cache.max_bytes()==seed,"1 GiB set after the change is the user's and stays");
    }
    setStored(root,seed);
    {
        vcr::DiskCache cache(root.string(),64);
        check(cache.max_bytes()==64,"an explicit limit wins over the old seed");
    }
    setStored(root,32);
    vcr::DiskCache cache(root.string(),-1);
    check(cache.max_bytes()==32,"a chosen limit on an older index stays");
}

int main() try {
    const auto root=fs::temp_directory_path()/("vcr-adopt-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {fs::path p;~Cleanup(){fs::remove_all(p);}} cleanup{root};
    const fs::path other=root.string()+"-unbounded";
    struct CleanupOther {fs::path p;~CleanupOther(){fs::remove_all(p);}} cleanupOther{other};
    unbounded(other);
    const fs::path fresh=root.string()+"-default";
    struct CleanupFresh {fs::path p;~CleanupFresh(){fs::remove_all(p);}} cleanupFresh{fresh};
    defaultBudget(fresh);
    const fs::path seeded=root.string()+"-seeded";
    struct CleanupSeeded {fs::path p;~CleanupSeeded(){fs::remove_all(p);}} cleanupSeeded{seeded};
    seededByOlderVersion(seeded);
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
