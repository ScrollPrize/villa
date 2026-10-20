#include "geometry/SharedGeometry.hpp"
#include "geometry/PersistentTree.hpp"
#include "geometry/ArrayBytes.hpp"
#include "internal/Spatial.hpp"

namespace vc::objects::detail {
using Json = nlohmann::json;
namespace {
constexpr int64_t blockSize = 64 * 1024;
std::vector<uint8_t> payload(sqlite3* db, const std::string& name, const std::string& hash) {
    Statement row(db, "SELECT data FROM geometry_chunks WHERE hash=?"); row.bind(1, hash);
    if (!row.row()) throw std::runtime_error("geometry: missing shared payload");
    auto* p = static_cast<const uint8_t*>(sqlite3_column_blob(row.value, 0));
    std::vector<uint8_t> bytes(p, p + sqlite3_column_bytes(row.value, 0));
    if (geometryWork) geometryWork->readBytes[name] += bytes.size();
    if (geometryPayloadHash(name, bytes) != hash) throw std::runtime_error("geometry: corrupt shared payload");
    return bytes;
}
void append(Json& out, Json part) {
    if (!part.at("length").get<int64_t>()) return;
    if (!out.empty() && out.back().at("hash") == part.at("hash") &&
        out.back().at("offset").get<int64_t>() + out.back().at("length").get<int64_t>() == part.at("offset").get<int64_t>())
    {
        out.back()["length"] = out.back().at("length").get<int64_t>() + part.at("length").get<int64_t>();
        out.back().erase("bounds"); out.back().erase("bounds_count");
    }
    else out.push_back(std::move(part));
}
}
bool identityManaged(sqlite3* db,const std::string& object) {
    Statement latest(db,"SELECT json_type(manifest,'$.sequences') FROM geometry_revisions WHERE object_uuid=? ORDER BY object_generation DESC,rowid DESC LIMIT 1");
    latest.bindBlob(1,parseUuid(object)); return latest.row() && !latest.isNull(0);
}
bool sharedArray(const Json& a) { return a.contains("tree") || a.contains("blocks") || a.contains("tiles"); }
std::string storeSharedPayload(sqlite3* db, const std::string& name, std::span<const uint8_t> bytes) {
    GeometryPhase phase("storeSharedPayload");
    const auto hash = geometryPayloadHash(name, bytes);
    Statement insert(db, "INSERT INTO geometry_chunks VALUES(?,?) ON CONFLICT(hash) DO NOTHING");
    insert.bind(1, hash); check(sqlite3_bind_blob64(insert.value, 2, bytes.data(), bytes.size(), SQLITE_TRANSIENT), db); insert.stepDone();
    if (geometryWork && sqlite3_changes(db)) geometryWork->writtenBytes[name] += bytes.size();
    Statement verify(db, "SELECT data FROM geometry_chunks WHERE hash=?"); verify.bind(1, hash); verify.row();
    if (geometryWork) geometryWork->readBytes[name] += sqlite3_column_bytes(verify.value, 0);
    if (size_t(sqlite3_column_bytes(verify.value, 0)) != bytes.size() ||
        (bytes.size() && std::memcmp(sqlite3_column_blob(verify.value, 0), bytes.data(), bytes.size())))
        throw std::runtime_error("geometry: corrupt or colliding shared payload");
    return hash;
}
Json sharedBlocks(sqlite3* db, const std::string& name, std::span<const uint8_t> bytes) {
    auto out = Json::array();
    for (size_t offset = 0; offset < bytes.size(); offset += blockSize) {
        const auto part = bytes.subspan(offset, std::min<size_t>(blockSize, bytes.size()-offset));
        out.push_back({{"hash", storeSharedPayload(db, name, part)}, {"offset", 0}, {"length", part.size()}});
    }
    return out;
}
Json sharedDescriptor(sqlite3* db, GeometryReader& reader, const std::string& name) {
    auto a = reader.manifest().at("arrays").at(name);
    if (sharedArray(a)) return a;
    const auto layout = geometryLayout(a);
    auto entries = Json::array();
    for (int64_t i = 0; i < layout.chunkCount(); ++i) {
        const std::string hash = a.at("chunks").at(i);
        Statement exists(db, "SELECT 1 FROM geometry_chunks WHERE hash=?"); exists.bind(1, hash);
        if (!exists.row()) storeSharedPayload(db, name, reader.chunk(name, i));
        if (layout.tile) {
            const auto [r,c,h,w] = layout.rectangle(i);
            entries.push_back({{"hash", hash}, {"row", r}, {"column", c}, {"rows", h}, {"columns", w}});
        } else entries.push_back({{"hash", hash}, {"offset", 0}, {"length", layout.chunkBytes(i)}});
    }
    a.erase("chunks"); a.erase("source_revision"); a.erase("storage");
    a[layout.tile ? "tiles" : "blocks"] = std::move(entries);
    if (layout.tile) a["tile_size"] = layout.tile;
    return a;
}
Json sliceSharedBlocks(const Json& blocks, int64_t offset, int64_t count) {
    if (offset < 0 || count < 0) throw std::invalid_argument("geometry: invalid block slice");
    auto result = Json::array();
    int64_t cursor = 0, copied = 0;
    for (const auto& b : blocks) {
        const auto size = b.at("length").get<int64_t>();
        if (size < 0 || cursor > INT64_MAX-size) throw std::runtime_error("geometry: invalid shared block length");
        const auto begin = std::max(offset, cursor), end = std::min(offset + count, cursor + size);
        if (begin < end) {
            auto part = b; part["offset"] = b.at("offset").get<int64_t>() + begin-cursor; part["length"] = end-begin;
            if (begin != cursor || end != cursor+size) { part.erase("bounds"); part.erase("bounds_count"); }
            append(result, std::move(part)); copied += end-begin;
        }
        cursor += size;
    }
    if (copied != count) throw std::out_of_range("geometry: shared slice out of range");
    if (!result.empty()) result.back().erase("bounds");
    return result;
}
std::vector<uint8_t> readSharedArray(sqlite3* db, const Json& a, const std::string& name, int64_t offset, int64_t count) {
    const auto length = a.at("byte_length").get<int64_t>();
    if (offset < 0 || count < 0 || offset > length || count > length-offset)
        throw std::out_of_range("geometry: shared read out of range");
    if (a.contains("tree") && !a.contains("tile_size")) return treeBytes(db,a.at("tree"),name,offset,count);
    if (a.contains("tree")) {
        const auto shape=a.at("shape").get<std::vector<int64_t>>();
        const auto rowBytes=shape[1]*surfacePixelBytes(a.at("dtype"),shape);
        std::vector<uint8_t> out; out.reserve(count);
        while (count) {
            const auto r=offset/rowBytes, start=offset%rowBytes, n=std::min(count,rowBytes-start);
            const auto pixel=surfacePixelBytes(a.at("dtype"),shape);
            const auto bytes=readSharedRectangle(db,a,name,r,start/pixel,1,(start%pixel+n+pixel-1)/pixel);
            out.insert(out.end(),bytes.begin()+start%pixel,bytes.begin()+start%pixel+n); offset+=n; count-=n;
        } return out;
    }
    std::vector<uint8_t> out(static_cast<size_t>(count));
    if (!count) return out;
    // Cache only chunks touched by this bounded request.
    std::map<std::string, std::vector<uint8_t>> loaded;
    auto get = [&](const Json& b) -> const std::vector<uint8_t>& {
        const std::string hash = b.at("hash"); auto found = loaded.find(hash);
        if (found == loaded.end()) found = loaded.emplace(hash, payload(db, name, hash)).first;
        return found->second;
    };
    if (a.contains("blocks")) {
        int64_t copied = 0;
        for (const auto& b : sliceSharedBlocks(a.at("blocks"), offset, count)) {
            const auto& bytes = get(b); const auto start = b.at("offset").get<int64_t>(), size = b.at("length").get<int64_t>();
            if (start < 0 || uint64_t(start) > bytes.size() || uint64_t(size) > bytes.size()-start)
                throw std::runtime_error("geometry: corrupt shared block range");
            std::memcpy(out.data()+copied, bytes.data()+start, size); copied += size;
        }
    } else {
        const auto shape = a.at("shape").get<std::vector<int64_t>>();
        const auto pixel = surfacePixelBytes(a.at("dtype"), shape), rowBytes = shape[1]*pixel;
        int64_t copied = 0;
        for (const auto& t : a.at("tiles")) {
            const auto r=t.at("row").get<int64_t>(), c=t.at("column").get<int64_t>(), h=t.at("rows").get<int64_t>(), w=t.at("columns").get<int64_t>();
            if (r < 0 || c < 0 || h <= 0 || w <= 0 || r > shape[0]-h || c > shape[1]-w)
                throw std::runtime_error("geometry: corrupt shared tile rectangle");
            for (auto y=std::max(r, offset/rowBytes); y<std::min(r+h, (offset+count-1)/rowBytes+1); ++y) {
                const auto base=y*rowBytes+c*pixel, begin=std::max(offset,base), end=std::min(offset+count,base+w*pixel);
                if (begin >= end) continue;
                const auto& bytes=get(t);
                if (int64_t(bytes.size()) != h*w*pixel) throw std::runtime_error("geometry: corrupt shared tile size");
                std::memcpy(out.data()+begin-offset, bytes.data()+(y-r)*w*pixel+begin-base,end-begin); copied+=end-begin;
            }
        }
        if (copied != count) throw std::runtime_error("geometry: incomplete shared tile coverage");
    }
    return out;
}
std::vector<uint8_t> readSharedRectangle(sqlite3* db, const Json& a, const std::string& name,
    int64_t row, int64_t column, int64_t rows, int64_t columns) {
    const auto shape=a.at("shape").get<std::vector<int64_t>>();
    if (row<0 || column<0 || rows<0 || columns<0 || row>shape[0] || column>shape[1] || rows>shape[0]-row || columns>shape[1]-column)
        throw std::out_of_range("geometry: invalid shared rectangle");
    const auto pixel=surfacePixelBytes(a.at("dtype"),shape);
    std::vector<uint8_t> out(rows*columns*pixel);
    if (!rows || !columns) return out;
    if (a.contains("tree") && a.contains("tile_size")) {
        const auto origin=a.at("origin").get<std::array<int64_t,2>>(); const int64_t tile=a.at("tile_size");
        auto floor=[&](int64_t x) { auto q=x/tile; if(x%tile<0)--q; return q*tile; };
        const auto top=row+origin[0],left=column+origin[1]; int64_t copied=0;
        for (auto y=floor(top);y<top+rows;y+=tile) for(auto x=floor(left);x<left+columns;x+=tile) {
            const auto t=treeGet(db,a.at("tree"),tileKey(y,x));
            if(t.is_null()) throw std::runtime_error("missing indexed surface tile");
            const int64_t r=t.at("row"),c=t.at("column"),h=t.at("rows"),w=t.at("columns");
            const auto rt=std::max(r,top),rb=std::min(r+h,top+rows),cl=std::max(c,left),cr=std::min(c+w,left+columns);
            const auto bytes=payload(db,name,t.at("hash"));
            if (int64_t(bytes.size())!=h*w*pixel || rt>=rb || cl>=cr) throw std::runtime_error("invalid indexed surface tile");
            for(auto yy=rt;yy<rb;++yy) std::memcpy(out.data()+((yy-top)*columns+cl-left)*pixel,bytes.data()+((yy-r)*w+cl-c)*pixel,(cr-cl)*pixel);
            copied+=(rb-rt)*(cr-cl)*pixel;
        }
        if(copied!=int64_t(out.size())) throw std::runtime_error("incomplete indexed surface rectangle");
    } else if (a.contains("blocks") || a.contains("tree")) {
        for (int64_t r=0;r<rows;++r) {
            const auto bytes=readSharedArray(db,a,name,((row+r)*shape[1]+column)*pixel,columns*pixel);
            std::memcpy(out.data()+r*columns*pixel,bytes.data(),bytes.size());
        }
    } else {
        int64_t copied=0;
        for (const auto& t:a.at("tiles")) {
            const int64_t r=t.at("row"),c=t.at("column"),h=t.at("rows"),w=t.at("columns");
            const auto top=std::max(row,r),bottom=std::min(row+rows,r+h),left=std::max(column,c),right=std::min(column+columns,c+w);
            if (top>=bottom || left>=right) continue;
            const auto bytes=payload(db,name,t.at("hash"));
            if (int64_t(bytes.size())!=h*w*pixel) throw std::runtime_error("geometry: corrupt shared tile size");
            for (auto y=top;y<bottom;++y) std::memcpy(out.data()+((y-row)*columns+left-column)*pixel,
                bytes.data()+((y-r)*w+left-c)*pixel,(right-left)*pixel);
            copied+=(bottom-top)*(right-left)*pixel;
        }
        if (copied!=int64_t(out.size())) throw std::runtime_error("geometry: incomplete shared tile coverage");
    }
    return out;
}
void normalizeSharedBlocks(sqlite3* db, const std::string& name, Json& a) {
    auto blocks = Json::array(); for (const auto& b : a.at("blocks")) append(blocks, b);
    a["blocks"] = std::move(blocks);
    // At most two blocks per 64 KiB plus 64 edit fragments. Compaction produces
    // direct payload references; no history chain is traversed by future reads.
    const auto limit = 64 + 2*(a.at("byte_length").get<int64_t>()/blockSize+1);
    if (a.at("blocks").size() > size_t(limit)) {
        if (geometryWork) ++geometryWork->compactions;
        const auto bytes=readSharedArray(db,a,name,0,a.at("byte_length"));
        a["blocks"]=sharedBlocks(db,name,bytes);
    }
}
void referenceSharedArrays(sqlite3* db, const std::string& revision, const Json& manifest) {
    Statement ref(db,"INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)");
    for (const auto& [name,a] : manifest.at("arrays").items()) {
        int64_t i=0;
        if (a.contains("tree")) {
            if (!a.at("tree").is_null()) {
                ref.reset(); ref.bindBlob(1,parseUuid(revision)); ref.bind(2,name); ref.bind(3,int64_t(0)); ref.bind(4,a.at("tree").at("hash").get<std::string>()); ref.stepDone();
                if (geometryWork) ++geometryWork->manifestEntries;
            }
            continue;
        }
        for (const auto& b : a.at(a.contains("blocks") ? "blocks" : "tiles")) {
            ref.reset(); ref.bindBlob(1,parseUuid(revision)); ref.bind(2,name); ref.bind(3,i++); ref.bind(4,b.at("hash").get<std::string>()); ref.stepDone();
            if (geometryWork) ++geometryWork->manifestEntries;
        }
    }
}
void editSharedArray(sqlite3* db, Json& manifest, const GeometryArrayEdit& edit) {
    auto& a=manifest.at("arrays").at(edit.array);
    const auto scalar=geometryByteLength(a.at("dtype"),{1}), length=a.at("byte_length").get<int64_t>();
    if (edit.elementOffset < 0 || edit.elementOffset > length/scalar || edit.bytes.size()%scalar || edit.bytes.size()>uint64_t(length-edit.elementOffset*scalar))
        throw std::invalid_argument("geometry: invalid shared edit range");
    if (edit.bytes.empty()) return;
    const auto offset=edit.elementOffset*scalar;
    if (a.contains("tree") && !a.contains("tile_size")) {
        const bool coordinates=edit.array=="points" || edit.array=="control_points" || edit.array=="line_points";
        const auto stride=coordinates?3*scalar:scalar;
        // Replacing scalar ranges includes complete boundary points in the new
        // leaf so cached bounds never depend on partially decoded coordinates.
        const auto begin=offset/stride*stride, end=(offset+int64_t(edit.bytes.size())+stride-1)/stride*stride;
        auto bytes=readSharedArray(db,a,edit.array,begin,end-begin);
        std::memcpy(bytes.data()+offset-begin,edit.bytes.data(),edit.bytes.size());
        const auto root=a.at("tree");
        a["tree"]=byteJoin(db,byteJoin(db,byteSlice(db,root,edit.array,a.at("dtype"),0,begin,coordinates),
            byteTree(db,edit.array,a.at("dtype"),bytes,coordinates),edit.array,a.at("dtype"),coordinates),
            byteSlice(db,root,edit.array,a.at("dtype"),end,length-end,coordinates),edit.array,a.at("dtype"),coordinates);
        const auto ids=edit.array=="control_points"?"control_point_ids":"point_ids";
        if (manifest.contains("sequences") && manifest.at("sequences").contains(edit.array) && manifest.at("arrays").contains(ids)) {
            auto& state=manifest["sequences"][edit.array];
            const auto idBytes=readSharedArray(db,manifest.at("arrays").at(ids),ids,begin/stride*8,(end-begin)/stride*8);
            std::vector<int64_t> values((end-begin)/stride); copyScalars<int64_t>(idBytes,values);
            for(size_t i=0;i<values.size();++i) {
                auto point=identityGet(db,state.at("identities"),values[i]);
                point["bytes"]=std::vector<uint8_t>(bytes.begin()+i*stride,bytes.begin()+(i+1)*stride);
                state["identities"]=identityPut(db,state.at("identities"),values[i],point);
            }
        }
    } else if (a.contains("tree")) {
        const auto shape=a.at("shape").get<std::vector<int64_t>>(); const auto pixel=surfacePixelBytes(a.at("dtype"),shape);
        const auto origin=a.at("origin").get<std::array<int64_t,2>>(); const int64_t tile=a.at("tile_size");
        auto floor=[&](int64_t x) {auto q=x/tile;if(x%tile<0)--q;return q*tile;};
        std::set<std::string> touched;
        for(int64_t pos=offset;pos<offset+int64_t(edit.bytes.size());) {
            const auto row=pos/(shape[1]*pixel), col=pos/pixel%shape[1];
            touched.insert(tileKey(floor(origin[0]+row),floor(origin[1]+col)));
            pos+=std::min(offset+int64_t(edit.bytes.size())-pos,std::min((shape[1]-col)*pixel-pos%pixel,(floor(origin[1]+col)+tile-origin[1]-col)*pixel-pos%pixel));
        }
        std::set<std::pair<int64_t,int64_t>> changedTiles;
        for(const auto& key:touched) {
            auto t=treeGet(db,a.at("tree"),key); auto bytes=payload(db,edit.array,t.at("hash"));
            const int64_t r=t.at("row").get<int64_t>()-origin[0],c=t.at("column").get<int64_t>()-origin[1],h=t.at("rows"),w=t.at("columns");
            for(auto y=r;y<r+h;++y) {
                const auto base=(y*shape[1]+c)*pixel,begin=std::max(offset,base),end=std::min(offset+int64_t(edit.bytes.size()),base+w*pixel);
                if(begin<end) std::memcpy(bytes.data()+(y-r)*w*pixel+begin-base,edit.bytes.data()+begin-offset,end-begin);
            }
            t["hash"]=storeSharedPayload(db,edit.array,bytes);
            t["bounds"]=edit.array=="points"?coordinateBounds(bytes,a.at("dtype"),true):Json();
            changedTiles.emplace(floor(r+origin[0]),floor(c+origin[1]));
            a["tree"]=treePut(db,a.at("tree"),key,treeLeaf(db,t,1,edit.array=="points"?coordinateBounds(bytes,a.at("dtype"),true):Json(),false,key));
        }
        if(edit.array=="points")a["tree"]=tileHalos(db,a.at("tree"),tile,changedTiles);
    } else if (a.contains("blocks")) {
        auto parts=sliceSharedBlocks(a.at("blocks"),0,offset);
        for (auto& b : sharedBlocks(db,edit.array,edit.bytes)) parts.push_back(std::move(b));
        for (auto& b : sliceSharedBlocks(a.at("blocks"),offset+edit.bytes.size(),length-offset-edit.bytes.size())) parts.push_back(std::move(b));
        a["blocks"]=std::move(parts); normalizeSharedBlocks(db,edit.array,a);
    } else {
        const auto shape=a.at("shape").get<std::vector<int64_t>>(); const auto pixel=surfacePixelBytes(a.at("dtype"),shape);
        const int64_t end=offset+edit.bytes.size();
        for (auto& t : a.at("tiles")) {
            const int64_t r=t.at("row"),c=t.at("column"),h=t.at("rows"),w=t.at("columns");
            std::vector<uint8_t> bytes;
            for (auto y=r;y<r+h;++y) {
                const auto base=(y*shape[1]+c)*pixel, begin=std::max(offset,base), stop=std::min(end,base+w*pixel);
                if (begin>=stop) continue;
                if (bytes.empty()) bytes=payload(db,edit.array,t.at("hash"));
                if (int64_t(bytes.size()) != h*w*pixel) throw std::runtime_error("geometry: corrupt shared tile");
                std::memcpy(bytes.data()+(y-r)*w*pixel+begin-base,edit.bytes.data()+begin-offset,stop-begin);
            }
            if (!bytes.empty()) { t["hash"]=storeSharedPayload(db,edit.array,bytes); t.erase("bounds"); }
        }
    }
}
}
