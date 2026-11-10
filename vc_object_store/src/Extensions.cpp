#include "internal/Extensions.hpp"
#include "internal/Database.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/RecordResolver.hpp"
#include <cctype>
#include <cmath>
#include <set>

namespace vc::objects {
using namespace detail;
namespace {
using Json = nlohmann::json;
std::string identifier(const std::string& name) {
    if(name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) || name[0]=='_') ||
       std::any_of(name.begin(),name.end(),[](unsigned char c){return !std::isalnum(c)&&c!='_';}))
        throw std::invalid_argument("extension: invalid SQL identifier: "+name);
    return "\""+name+"\"";
}
const char* sqlType(ExtensionType type) {
    switch(type) {
        case ExtensionType::Integer:return "INTEGER";
        case ExtensionType::Real:return "REAL";
        case ExtensionType::Text:return "TEXT";
        case ExtensionType::Blob:return "BLOB";
    }
    throw std::invalid_argument("extension: invalid SQL type");
}
ExtensionType columnType(const std::string& type) {
    for(auto t:{ExtensionType::Integer,ExtensionType::Real,ExtensionType::Text,ExtensionType::Blob})
        if(type==sqlType(t))return t;
    throw std::runtime_error("extension: unsupported column type: "+type);
}
std::string joined(std::span<const std::string> names) {
    std::string out;
    for(const auto& name:names){if(!out.empty())out+=",";out+=identifier(name);}
    return out;
}
void registry(sqlite3* db) {
    exec(db,R"SQL(CREATE TABLE IF NOT EXISTS extension_tables(
 table_uuid BLOB PRIMARY KEY REFERENCES table_directory(uuid),
 extension TEXT NOT NULL, format_version INTEGER NOT NULL CHECK(format_version>0),
 based_on TEXT NOT NULL CHECK(json_valid(based_on) AND json_type(based_on)='array'),
 provisioning TEXT NOT NULL CHECK(json_valid(provisioning) AND json_type(provisioning)='object'),
 key_space TEXT NOT NULL CHECK(key_space IN ('','objects','edges')),
 versioned INTEGER NOT NULL CHECK(versioned IN (0,1)),
 last_modified_commit INTEGER NOT NULL CHECK(last_modified_commit>=0)
);)SQL");
}
void changed(sqlite3* db,const std::string& table) {
    const auto sequence=writeCommitSequence(db);
    execute(db,"UPDATE extension_tables SET last_modified_commit=? WHERE table_uuid="
        "(SELECT uuid FROM table_directory WHERE name=?) AND last_modified_commit<>?",sequence,table,sequence);
    exec(db,"UPDATE store_write_state SET dirty=1 WHERE singleton=1 AND dirty=0;");
    // The record-only feed cannot express arbitrary extension rows. Invalidate
    // older cursors rather than pretend that an extension-only commit was empty.
    execute(db,"UPDATE store_manifest SET feed_start=? WHERE singleton=1 AND feed_start<?",sequence,sequence);
}
void bindValue(Statement& statement,int col,const ExtensionValue& value) {
    std::visit([&](const auto& v){using T=std::decay_t<decltype(v)>;
        if constexpr(std::is_same_v<T,std::monostate>)statement.bindNull(col);
        else if constexpr(std::is_same_v<T,std::vector<uint8_t>>) {
            if(v.empty())check(sqlite3_bind_zeroblob(statement.value,col,0),sqlite3_db_handle(statement.value));
            else statement.bind(col,std::span<const uint8_t>(v));
        }
        else statement.bind(col,v);
    },value);
}
ExtensionRow readRow(const Statement& rows) {
    ExtensionRow row;row.reserve(sqlite3_column_count(rows.value));
    for(int col=0;col<sqlite3_column_count(rows.value);++col)switch(sqlite3_column_type(rows.value,col)) {
        case SQLITE_NULL:row.emplace_back(std::monostate{});break;
        case SQLITE_INTEGER:row.emplace_back(rows.i64(col));break;
        case SQLITE_FLOAT:row.emplace_back(rows.f64(col));break;
        case SQLITE_TEXT:row.emplace_back(rows.text(col));break;
        case SQLITE_BLOB:{const auto* p=static_cast<const uint8_t*>(sqlite3_column_blob(rows.value,col));
            const auto n=sqlite3_column_bytes(rows.value,col);row.emplace_back(n?std::vector<uint8_t>(p,p+n):std::vector<uint8_t>{});break;}
    }
    return row;
}
void validateValue(const ExtensionColumn& column,const ExtensionValue& value) {
    if(std::holds_alternative<std::monostate>(value)) {
        if(!column.nullable)throw std::invalid_argument("extension: NULL in required column "+column.name);
        return;
    }
    const bool valid=column.type==ExtensionType::Integer?std::holds_alternative<int64_t>(value):
        column.type==ExtensionType::Real?std::holds_alternative<double>(value):
        column.type==ExtensionType::Text?std::holds_alternative<std::string>(value):std::holds_alternative<std::vector<uint8_t>>(value);
    if(!valid)throw std::invalid_argument("extension: wrong value type for "+column.name);
    if(const auto* real=std::get_if<double>(&value);real&&!std::isfinite(*real))
        throw std::invalid_argument("extension: nonfinite real value");
}
void insertRows(sqlite3* db,const ExtensionTableInfo& info,std::span<const ExtensionRow* const> rows) {
    if(rows.empty())return;
    std::string parameters;
    for(size_t i=0;i<info.columns.size();++i){if(i)parameters+=",";parameters+="?";}
    Statement insert(db,"INSERT INTO "+identifier(info.name)+" VALUES("+parameters+")");
    size_t uuidColumn=0;
    if(!info.keySpace.empty()) {
        const auto it=std::find_if(info.columns.begin(),info.columns.end(),[&](const auto& c){return c.name==info.primaryKey.front();});
        uuidColumn=size_t(it-info.columns.begin());
    }
    for(const auto* row:rows) {
        if(!row || row->size()!=info.columns.size())throw std::invalid_argument("extension: row column count differs");
        insert.reset();
        for(size_t i=0;i<row->size();++i){validateValue(info.columns[i],(*row)[i]);bindValue(insert,int(i+1),(*row)[i]);}
        if(!info.keySpace.empty()&&std::get<std::vector<uint8_t>>((*row)[uuidColumn]).size()!=16)
            throw std::invalid_argument("extension: routing key must be a binary UUID");
        insert.stepDone();
    }
    changed(db,info.name);
}
}

std::string extensionSourcesJson(std::span<const ExtensionSource> sources) {
    auto json=Json::array();std::set<std::pair<std::string,std::string>> seen;
    for(const auto& source:sources){
        const auto uuid=util::canonicalUuid(source.storeUuid),generation=util::canonicalUuid(source.generation);
        if(source.commitSequence<0 || !seen.emplace(uuid,generation).second)
            throw std::invalid_argument("extension: invalid or repeated input state");
        json.push_back({{"store_uuid",uuid},{"generation",generation},{"commit_sequence",source.commitSequence}});
    }
    return json.dump();
}
std::vector<ExtensionSource> extensionSourcesFromJson(const std::string& text) {
    const auto json=Json::parse(text);if(!json.is_array())throw std::invalid_argument("extension: input states must be an array");
    std::vector<ExtensionSource> out;
    for(const auto& s:json){
        if(!s.is_object()||s.size()!=3||!s.at("commit_sequence").is_number_integer())
            throw std::invalid_argument("extension: invalid input state");
        out.push_back({util::canonicalUuid(s.at("store_uuid").get<std::string>()),
            util::canonicalUuid(s.at("generation").get<std::string>()),s.at("commit_sequence").get<int64_t>()});
    }
    (void)extensionSourcesJson(out);return out;
}
ExtensionValue extensionUuid(const std::string& text){const auto bytes=parseUuid(text);return std::vector<uint8_t>(bytes.begin(),bytes.end());}
std::string extensionUuid(const ExtensionValue& value){const auto& b=std::get<std::vector<uint8_t>>(value);
    if(b.size()!=16)throw std::invalid_argument("extension: invalid UUID bytes");
    std::array<unsigned char,16> bytes;std::copy(b.begin(),b.end(),bytes.begin());return formatUuid(bytes);}
size_t extensionPayloadBytes(std::span<const InitialExtensionRow> rows){size_t bytes=0;
    for(const auto& row:rows){bytes+=sizeof(row)+row.table.size()+row.values.size()*sizeof(ExtensionValue);
        for(const auto& v:row.values)std::visit([&](const auto& x){if constexpr(requires{x.size();})bytes+=x.size();},v);}
    return bytes;}

bool detail::hasExtensionTables(sqlite3* db) {
    Statement present(db,"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='extension_tables'");
    if(!present.row())return false;
    Statement rows(db,"SELECT 1 FROM extension_tables LIMIT 1");return rows.row();
}
std::optional<ExtensionTableInfo> detail::extensionTable(sqlite3* db,const std::string& name,bool includeMetadata) {
    if(!hasExtensionTables(db))return {};
    // Batch insertion/row scans need the layout, not dataset-level provenance.
    // In particular, do not copy a large source selection list per batch.
    const auto metadata=includeMetadata?"e.based_on,e.provisioning":"'[]','{}'";
    Statement row(db,std::string("SELECT d.uuid,d.name,e.extension,e.format_version,")+metadata+",e.key_space,e.last_modified_commit,e.versioned "
        "FROM extension_tables e JOIN table_directory d ON e.table_uuid=d.uuid WHERE d.name=?",name);
    if(!row.row())return {};
    ExtensionTableInfo info;info.tableUuid=row.uuid(0);info.name=row.text(1);info.extension=row.text(2);
    info.formatVersion=row.i64(3);info.basedOn=extensionSourcesFromJson(row.text(4));info.provisioning=row.text(5);
    info.keySpace=row.text(6);info.lastModifiedCommit=row.i64(7);
    info.versioned=row.i64(8)!=0;
    std::map<int64_t,std::string> keys;
    Statement columns(db,"SELECT name,type,\"notnull\",pk FROM pragma_table_info(?) ORDER BY cid",name);
    while(columns.row()){info.columns.push_back({columns.text(0),columnType(columns.text(1)),columns.i64(2)==0});
        if(columns.i64(3))keys.emplace(columns.i64(3),columns.text(0));}
    for(const auto& [i,key]:keys)info.primaryKey.push_back(key);
    if(includeMetadata){Statement indexes(db,"SELECT name,\"unique\" FROM pragma_index_list(?) WHERE origin='c' ORDER BY name",name);
    while(indexes.row()) {ExtensionIndex index;index.unique=indexes.i64(1)!=0;
        Statement indexed(db,"SELECT name FROM pragma_index_info(?) ORDER BY seqno",indexes.text(0));
        while(indexed.row())index.columns.push_back(indexed.text(0));
        info.indexes.push_back(std::move(index));}}
    return info;
}
std::vector<ExtensionTableInfo> detail::extensionTables(sqlite3* db) {
    std::vector<ExtensionTableInfo> out;if(!hasExtensionTables(db))return out;
    Statement rows(db,"SELECT d.name FROM extension_tables e JOIN table_directory d ON d.uuid=e.table_uuid ORDER BY d.name");
    while(rows.row())out.push_back(*extensionTable(db,rows.text(0)));return out;
}
void detail::createExtensionTable(sqlite3* db,const ExtensionTable& spec) {
    if(!spec.name.starts_with("ext_") || spec.extension.empty()||spec.formatVersion<1||spec.columns.empty()||spec.primaryKey.empty())
        throw std::invalid_argument("extension: name, identifier, format, columns and primary key are required");
    const auto name=identifier(spec.name),sources=extensionSourcesJson(spec.basedOn);
    const auto provisioning=Json::parse(spec.provisioning);
    if(!provisioning.is_object())throw std::invalid_argument("extension: provisioning must be a JSON object");
    if(!spec.keySpace.empty()&&spec.keySpace!="objects"&&spec.keySpace!="edges")throw std::invalid_argument("extension: unsupported routing space");
    if(spec.versioned&&spec.keySpace.empty())throw std::invalid_argument("extension: versioned fields require an owner key space");
    std::set<std::string> names,keys;std::string body;
    for(const auto& column:spec.columns){
        if(!names.insert(column.name).second)throw std::invalid_argument("extension: duplicate column");
        if(!body.empty())body+=",";
        body+=identifier(column.name)+" "+sqlType(column.type)+(column.nullable?"":" NOT NULL");
    }
    for(const auto& key:spec.primaryKey){
        if(!names.contains(key)||!keys.insert(key).second)throw std::invalid_argument("extension: invalid primary key");
        const auto c=std::find_if(spec.columns.begin(),spec.columns.end(),[&](const auto& v){return v.name==key;});
        if(c->nullable)throw std::invalid_argument("extension: primary key must not be nullable");
    }
    if(!spec.keySpace.empty()) {
        const auto c=std::find_if(spec.columns.begin(),spec.columns.end(),[&](const auto& v){return v.name==spec.primaryKey.front();});
        if(c->type!=ExtensionType::Blob)throw std::invalid_argument("extension: UUID routing requires BLOB key");
    }
    for(const auto& index:spec.indexes){if(index.columns.empty())throw std::invalid_argument("extension: empty index");
        for(const auto& key:index.columns)if(!names.contains(key))throw std::invalid_argument("extension: unknown index column");}
    if(auto old=extensionTable(db,spec.name)) {
        if(old->extension!=spec.extension||old->formatVersion!=spec.formatVersion||old->columns!=spec.columns||
            old->primaryKey!=spec.primaryKey||old->keySpace!=spec.keySpace||old->versioned!=spec.versioned||Json::parse(old->provisioning)!=provisioning||
            extensionSourcesJson(old->basedOn)!=sources)throw std::invalid_argument("extension: registered definition/provenance differs");
        if(old->indexes.size()!=spec.indexes.size())throw std::invalid_argument("extension: index count differs");
        for(size_t i=0;i<spec.indexes.size();++i){const auto& idx=spec.indexes[i];
            Statement exists(db,"SELECT sql FROM sqlite_schema WHERE type='index' AND name=?",spec.name+"_idx"+std::to_string(i));
            const auto expected=std::string("CREATE ")+(idx.unique?"UNIQUE ":"")+"INDEX "+identifier(spec.name+"_idx"+std::to_string(i))+" ON "+name+"("+joined(idx.columns)+")";
            if(!exists.row()||exists.text(0)!=expected)throw std::invalid_argument("extension: index definition differs");}
        return;
    }
    Savepoint save(db);registry(db);
    exec(db,("CREATE TABLE "+name+"("+body+",PRIMARY KEY("+joined(spec.primaryKey)+")) WITHOUT ROWID").c_str());
    for(size_t i=0;i<spec.indexes.size();++i){const auto& idx=spec.indexes[i];
        exec(db,(std::string("CREATE ")+(idx.unique?"UNIQUE ":"")+"INDEX "+identifier(spec.name+"_idx"+std::to_string(i))+" ON "+name+"("+joined(idx.columns)+")").c_str());}
    registerStorageTables(db);
    execute(db,"INSERT INTO extension_tables SELECT uuid,?,?,?,?,?,?,? FROM table_directory WHERE name=?",
        spec.extension,spec.formatVersion,sources,provisioning.dump(),spec.keySpace,int64_t(spec.versioned),writeCommitSequence(db),spec.name);
    changed(db,spec.name);save.release();
}
void detail::appendExtensionRows(sqlite3* db,const std::string& name,std::span<const ExtensionRow> rows) {
    const auto info=extensionTable(db,name,false);if(!info)throw std::invalid_argument("extension: unregistered table "+name);
    if(info->versioned)throw std::invalid_argument("extension: versioned fields require replaceExtensionRows");
    if(rows.empty())return;
    Savepoint save(db);std::vector<const ExtensionRow*> pointers;pointers.reserve(rows.size());
    for(const auto& row:rows)pointers.push_back(&row);
    insertRows(db,*info,pointers);save.release();
}
void detail::replaceCurrentExtensionRows(sqlite3* db,const std::string& name,const std::string& ownerUuid,std::span<const ExtensionRow> rows) {
    const auto info=extensionTable(db,name);
    if(!info||!info->versioned||info->keySpace.empty())throw std::invalid_argument("extension: replacement requires a versioned owner table");
    const auto owner=extensionUuid(ownerUuid);
    const auto key=std::find_if(info->columns.begin(),info->columns.end(),[&](const auto& c){return c.name==info->primaryKey.front();});
    const auto ordinal=size_t(key-info->columns.begin());
    std::vector<const ExtensionRow*> pointers;
    for(const auto& row:rows){if(row.size()!=info->columns.size()||row[ordinal]!=owner)
        throw std::invalid_argument("extension: replacement row belongs to another owner");pointers.push_back(&row);}
    Savepoint save(db);Statement erase(db,"DELETE FROM "+identifier(name)+" WHERE "+identifier(info->primaryKey.front())+"=?");
    bindValue(erase,1,owner);erase.stepDone();insertRows(db,*info,pointers);changed(db,name);save.release();
}
void detail::appendInitialExtensions(sqlite3* db,std::span<const InitialExtensionRow* const> rows) {
    std::map<std::string,std::vector<const ExtensionRow*>> tables;
    for(const auto* row:rows)tables[row->table].push_back(&row->values);
    for(const auto& [name,values]:tables){const auto info=extensionTable(db,name,false);
        if(!info)throw std::invalid_argument("extension: unregistered initial table "+name);
        if(info->versioned)throw std::invalid_argument("extension: initial rows must not bypass owner versioning");
        insertRows(db,*info,values);}
}
void detail::scanExtensionRows(sqlite3* db,const std::string& name,const std::function<bool(const ExtensionRow&)>& visitor,
                               std::span<const ExtensionValue> prefix) {
    const auto info=extensionTable(db,name,false);if(!info)throw std::invalid_argument("extension: unregistered table "+name);
    if(!visitor||prefix.size()>info->primaryKey.size())throw std::invalid_argument("extension: invalid scan");
    std::string where;
    for(size_t i=0;i<prefix.size();++i){where+=i?" AND ":" WHERE ";where+=identifier(info->primaryKey[i])+"=?";}
    Statement rows(db,"SELECT * FROM "+identifier(name)+where+" ORDER BY "+joined(info->primaryKey));
    for(size_t i=0;i<prefix.size();++i)bindValue(rows,int(i+1),prefix[i]);
    while(rows.row())if(!visitor(readRow(rows)))break;
}
void detail::rejectExtensionLoss(sqlite3* db,const char* operation){if(hasExtensionTables(db))
    throw std::runtime_error(std::string(operation)+": registered extension tables require a database-preserving copy; refusing silent data loss");}

ExtensionTable detail::extensionDefinition(const ExtensionTableInfo& info){
    return {info.name,info.extension,info.formatVersion,info.provisioning,info.basedOn,info.columns,
        info.primaryKey,info.indexes,info.keySpace,info.versioned};
}
void detail::requireCompatibleExtension(const ExtensionTableInfo& a,const ExtensionTableInfo& b){
    if(a.extension!=b.extension||a.formatVersion!=b.formatVersion||a.columns!=b.columns||
       a.primaryKey!=b.primaryKey||a.keySpace!=b.keySpace||a.versioned!=b.versioned||
       Json::parse(a.provisioning)!=Json::parse(b.provisioning))
        throw std::runtime_error("incompatible extension definitions: "+a.name);
}
void ObjectStore::createExtensionTable(WriteTxn& t,const ExtensionTable& spec){detail::createExtensionTable(static_cast<sqlite3*>(connection(t)),spec);}
std::vector<ExtensionTableInfo> ObjectStore::extensionTables(const Txn& t) const{
    std::vector<ExtensionTableInfo> result;
    for(auto* source:querySources(static_cast<sqlite3*>(connection(t)))){
        auto tables=detail::extensionTables(source);result.insert(result.end(),tables.begin(),tables.end());
    }
    return result;
}
void ObjectStore::appendExtensionRows(WriteTxn& t,const std::string& name,std::span<const ExtensionRow> rows){detail::appendExtensionRows(static_cast<sqlite3*>(connection(t)),name,rows);}
void ObjectStore::scanExtensionRows(const Txn& t,const std::string& name,const std::function<bool(const ExtensionRow&)>& visitor,std::span<const ExtensionValue> prefix) const {
    auto* db=static_cast<sqlite3*>(connection(t));
    std::optional<ExtensionTableInfo> table;
    const auto sources=querySources(db);
    for(auto* source:sources)if(auto candidate=extensionTable(source,name)){
        if(table)requireCompatibleExtension(*table,*candidate);else table=std::move(candidate);
    }
    if(!table)throw std::invalid_argument("extension: unregistered table "+name);
    if(!visitor||prefix.size()>table->primaryKey.size())throw std::invalid_argument("extension: invalid scan");
    if(table->versioned){
        std::set<std::string> seen;bool more=true;
        for(auto* source:sources){
            scanExtensionOwnerKeys(source,*table,prefix,[&](const std::string& owner){
                if(!seen.insert(owner).second)return true;
                std::optional<std::string> meta;
                if(table->keySpace=="objects") {if(auto record=objectByUuid(t,owner))meta=record->meta;}
                else if(auto record=edgeByUuid(t,owner))meta=record->meta;
                if(meta)more=scanExtensionSnapshot(db,*table,owner,*meta,visitor,prefix);
                return more;
            });
            if(!more)return;
        }
        return;
    }
    std::vector<size_t> keys;
    for(const auto& key:table->primaryKey)keys.push_back(std::find_if(table->columns.begin(),table->columns.end(),
        [&](const auto& c){return c.name==key;})-table->columns.begin());
    std::set<ExtensionRow> seen;bool more=true;
    for(auto* source:sources){
        if(!extensionTable(source,name))continue;
        detail::scanExtensionRows(source,name,[&](const ExtensionRow& row){
            ExtensionRow key;for(auto i:keys)key.push_back(row.at(i));
            if(seen.insert(std::move(key)).second)more=visitor(row);return more;
        },prefix);
        if(!more)return;
    }
}
} // namespace vc::objects
