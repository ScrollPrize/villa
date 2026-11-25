#pragma once
#include "internal/RecordResolver.hpp"
#include <algorithm>
#include <queue>

namespace vc::objects::detail {
// Physical readers retain their indexed execution plans. Logical visibility
// precedes global ordering/pagination, for objects, regions and relationships.
template<class Read,class Visible,class Less>
auto selectVisibleRecords(sqlite3* db,Read read,Visible visible,Less less,
                          int64_t offset=0,std::optional<int64_t> limit=std::nullopt)
{
    decltype(read(db)) result;
    for(auto* source:querySources(db))for(auto& record:read(source))
        if(visible(record,source))result.push_back(std::move(record));
    std::sort(result.begin(),result.end(),less);
    const auto begin=std::min(result.size(),size_t(offset));
    const auto end=limit?begin+std::min(result.size()-begin,size_t(*limit)):result.size();
    result.erase(result.begin()+end,result.end());
    result.erase(result.begin(),result.begin()+begin);
    return result;
}
// Merge indexed physical queries in UUID order. The selected key column is the object
// UUID; rows within each object retain their physical query's ordering. All
// shadowing/deletion decisions use the same resolver as point lookups.
class VisibleObjectRows {
    struct Cursor {
        sqlite3* source;
        std::unique_ptr<Statement> rows;
        std::optional<std::string> key;
    };
    sqlite3* root_;
    int keyColumn_;
    std::vector<Cursor> cursors_;
    struct Next {
        const std::string* key;
        size_t cursor;
    };
    struct Later {
        bool operator()(const Next& a,const Next& b)const{
            return *a.key==*b.key?a.cursor>b.cursor:*a.key>*b.key;
        }
    };
    std::priority_queue<Next,std::vector<Next>,Later> next_;
    std::optional<size_t> current_;
    void advance(Cursor& cursor) {
        cursor.key.reset();
        if(cursor.rows->row())cursor.key=cursor.rows->uuid(keyColumn_);
    }
    void advance(size_t cursor){
        advance(cursors_[cursor]);
        if(cursors_[cursor].key)next_.push({&*cursors_[cursor].key,cursor});
    }
public:
    template<class Query> VisibleObjectRows(sqlite3* root,Query query,int keyColumn=0):root_(root),keyColumn_(keyColumn) {
        for(auto* source:querySources(root)){
            auto rows=query(source);
            if(!rows)continue;
            cursors_.push_back({source,std::move(rows),{}});
        }
        // Finish growing cursors_ before storing pointers to its keys. Each
        // indexed cursor then owns at most one heap entry until advanced.
        for(size_t i=0;i<cursors_.size();++i)advance(i);
    }
    bool row() {
        if(current_)advance(*current_);
        for(;;) {
            current_.reset();
            if(next_.empty())return false;
            current_=next_.top().cursor;next_.pop();
            auto& cursor=cursors_[*current_];
            if(objectSource(root_,*cursor.key,false,false,cursor.source)==cursor.source)return true;
            advance(*current_);
        }
    }
    Statement& values(){return *cursors_.at(current_.value()).rows;}
    sqlite3* source()const{return cursors_.at(current_.value()).source;}
    const std::string& key()const{return *cursors_.at(current_.value()).key;}
};

inline std::optional<std::string> objectUuidFilter(const std::optional<std::vector<std::string>>& uuids)
{
    if(!uuids)return {};
    // SQLite hex() yields uppercase; compare UUID bytes, not spelling/case.
    auto values=nlohmann::json::array();
    constexpr char digits[]="0123456789ABCDEF";
    for(const auto& uuid:*uuids){
        std::string value;value.reserve(32);
        for(auto byte:parseUuid(uuid)){value+=digits[byte>>4];value+=digits[byte&15];}
        values.push_back(std::move(value));
    }
    return values.dump();
}
}
