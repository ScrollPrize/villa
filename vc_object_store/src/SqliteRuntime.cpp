#include "vc_object_store/SqliteRuntime.hpp"
#include <sqlite3.h>
#include <limits>
#include <stdexcept>

namespace vc::objects {
void configureSqliteRuntime(std::optional<bool> memoryStatistics,std::size_t pageCacheBulkBytes)
{
    const auto kib=pageCacheBulkBytes/1024+(pageCacheBulkBytes%1024!=0);
    if(kib>std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("SQLite page-cache bulk allocation is too large");
    const auto requireStartup=[](int rc){
        if(rc!=SQLITE_OK)
            throw std::runtime_error("SQLite runtime options must be configured before SQLite initialization; active connections are never shut down or reconfigured");
    };
    if(memoryStatistics)requireStartup(sqlite3_config(SQLITE_CONFIG_MEMSTATUS,int(*memoryStatistics)));
    if(pageCacheBulkBytes)
        requireStartup(sqlite3_config(SQLITE_CONFIG_PAGECACHE,nullptr,0,-int(kib)));
}
}
