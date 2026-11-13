#include "TestSupport.hpp"
#include "internal/Database.hpp"
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace {
bool accepts(sqlite3* db,const std::string& sql)
{
    char* error=nullptr;
    const auto result=sqlite3_exec(db,sql.c_str(),nullptr,nullptr,&error);
    const std::string message=error?error:"";sqlite3_free(error);
    if(result==SQLITE_OK)return true;
    if((result&0xff)==SQLITE_CONSTRAINT)return false;
    throw std::runtime_error("unexpected SQL failure: "+message);
}
void check(HistoryMode mode)
{
    TemporaryDirectory dir;OpenOptions options;options.author="test";options.history=mode;
    auto store=ObjectStore::create(dir.path/"approval.vcobj",options);auto txn=store.write();
    auto* db=ObjectStoreImplAccess::conn(txn);
    ObjectRecord a,b;a.uuid=util::newUuid();a.kind="note";b=a;b.uuid=util::newUuid();
    store.upsertObject(txn,a);store.upsertObject(txn,b);
    EdgeRecord edge;edge.aUuid=a.uuid;edge.bUuid=b.uuid;edge.kind="test";store.addEdge(txn,edge);
    exec(db,"CREATE TEMP TABLE old_approval(value TEXT NOT NULL CHECK(value IN ('pending','approved','rejected')));"
        "CREATE TEMP TABLE new_approval(value TEXT NOT NULL CHECK(value='pending' OR value='approved' OR value='rejected')); ");
    const std::vector<std::pair<std::string,bool>> cases={
        {"'pending'",true},{"'approved'",true},{"'rejected'",true},
        {"NULL",false},{"''",false},{"'Pending'",false},{"'APPROVED'",false},
        {"'rejected '",false},{"' pending'",false},{"'pending'||char(10)",false},
        {"'pending'||char(0)",false},{"char(0)||'pending'",false},
        {"'approved'||char(160)",false},{"'pendingx'",false},
        {"0",false},{"1",false},{"-1",false},{"1.25",false},{"TRUE",false},
        {"'0'",false},{"'1'",false},{"X'70656e64696e67'",false},
        {"CAST(X'70656e64696e67' AS TEXT)",true},{"'pend'||'ing'",true}
    };
    for(const auto& [value,expected]:cases){
        Savepoint trial(db);
        const auto legacy=accepts(db,"INSERT INTO old_approval VALUES("+value+")");
        const auto replacement=accepts(db,"INSERT INTO new_approval VALUES("+value+")");
        require(legacy==expected&&replacement==legacy,"approval CHECK accepts exactly the old domain");
        for(const auto* table:{"objects","edges"})
            require(accepts(db,std::string("UPDATE ")+table+" SET approval_state="+value)==legacy,
                "native approval checks agree with legacy CHECK including affinity and NOT NULL");
    }
    // Isolate the CHECK from all object/edge indexes and foreign-key programs.
    // Do not assume the legacy IN implementation always uses an ephemeral
    // table: a future SQLite release may optimize that too.
    Statement program(db,"EXPLAIN INSERT INTO new_approval VALUES(?)");
    while(program.row()) require(program.text(1)!="OpenEphemeral","OR approval CHECK must not construct a per-row temporary B-tree");
    for(const auto* table:{"objects","edges"}) {
        Statement schema(db,"SELECT sql FROM sqlite_schema WHERE name=?",std::string(table));
        require(schema.row()&&schema.text(0).find("approval_state='pending' OR")!=std::string::npos,
            "new native tables use direct comparisons");
    }
    // No publication is needed: raw SQL here deliberately isolates the schema
    // constraint rather than manufacturing approval provenance/history events.
}
}
int main(){try{check(HistoryMode::On);check(HistoryMode::Off);
    std::cout<<"approval CHECK parity passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
