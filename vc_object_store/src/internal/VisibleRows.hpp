#pragma once
#include "internal/RecordResolver.hpp"

namespace vc::objects::detail {
// Merge indexed physical queries in UUID order. Column zero is the object
// UUID; rows within each object retain their physical query's ordering. All
// shadowing/deletion decisions use the same resolver as point lookups.
class VisibleObjectRows {
    struct Cursor {
        sqlite3* source;
        std::unique_ptr<Statement> rows;
        std::optional<std::string> key;
    };
    sqlite3* root_;
    std::vector<Cursor> cursors_;
    std::optional<size_t> current_;
    void advance(Cursor& cursor) {
        cursor.key.reset();
        while(cursor.rows->row()) {
            auto key=cursor.rows->uuid(0);
            if(objectSource(root_,key)==cursor.source){cursor.key=std::move(key);break;}
        }
    }
public:
    template<class Query> VisibleObjectRows(sqlite3* root,Query query):root_(root) {
        for(auto* source:querySources(root)){
            auto rows=query(source);
            if(!rows)continue;
            cursors_.push_back({source,std::move(rows),{}});
            advance(cursors_.back());
        }
    }
    bool row() {
        if(current_)advance(cursors_[*current_]);
        current_.reset();
        for(size_t i=0;i<cursors_.size();++i)
            if(cursors_[i].key&&(!current_||*cursors_[i].key<*cursors_[*current_].key))current_=i;
        return current_.has_value();
    }
    Statement& values(){return *cursors_.at(current_.value()).rows;}
    sqlite3* source()const{return cursors_.at(current_.value()).source;}
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
