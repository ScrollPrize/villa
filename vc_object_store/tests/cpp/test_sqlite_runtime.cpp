#include "vc_object_store/SqliteRuntime.hpp"
#include <sqlite3.h>
#include <iostream>
#include <limits>
#include <stdexcept>

int main()
{
    try{
        bool rejected=false;
        try{vc::objects::configureSqliteRuntime({},std::numeric_limits<size_t>::max());}
        catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("overflowing startup allocation was accepted");
        vc::objects::configureSqliteRuntime(false,1024*1024);
        sqlite3* db=nullptr;
        if(sqlite3_open(":memory:",&db)!=SQLITE_OK)throw std::runtime_error("memory test connection failed");
        const auto close=[&]{sqlite3_close(db);};
        int current=0,peak=0;
        if(sqlite3_status(SQLITE_STATUS_MEMORY_USED,&current,&peak,0)!=SQLITE_OK||current!=0||peak!=0){
            close();throw std::runtime_error("SQLite global memory statistics were not disabled");
        }
        rejected=false;
        try{vc::objects::configureSqliteRuntime(false);}
        catch(const std::runtime_error&){rejected=true;}
        if(!rejected){close();throw std::runtime_error("active SQLite runtime was reconfigured");}
        vc::objects::configureSqliteRuntime(); // Unspecified options are a no-op.
        if(sqlite3_exec(db,"CREATE TABLE data(value); INSERT INTO data VALUES(42);",nullptr,nullptr,nullptr)!=SQLITE_OK){
            close();throw std::runtime_error("rejected reconfiguration damaged an active connection");
        }
        close();return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
