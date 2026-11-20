#pragma once
#include "geometry/GeometryReader.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
namespace vc::objects::detail {
// The rows [begin,end) of the array a fiber or track's length is measured
// along. A fiber with line points uses its annotated line (no tails, see
// vc::geometry::annotatedLineRange), else its control polyline. `rows(name)`
// returns an array's row count; `read(name, byteOffset, byteCount)` its bytes.
struct LengthRows { std::string name; int64_t begin = 0, end = 0; };
template<class Rows, class Read>
LengthRows lengthRows(const std::string& kind, const Rows& rows, const Read& read) {
    if (kind != "fiber") return {"points", 0, rows("points")};
    const auto lines = rows("line_points"), controls = rows("control_points");
    if (!lines) return {"control_points", 0, controls};
    if (controls < 2) return {"line_points", 0, lines};
    auto points = [&](const char* name, int64_t count) {
        const auto bytes = read(name, 0, count * 24);
        const std::span<const uint8_t> view(bytes);
        std::vector<vc::geometry::Point> result(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) copyScalars<double>(view.subspan(i * 24, 24), result[size_t(i)]);
        return result;
    };
    const auto range = vc::geometry::annotatedLineRange(points("control_points", controls), points("line_points", lines));
    return {"line_points", int64_t(range.begin), int64_t(range.end)};
}
// Length of the polyline of rows [begin,end) (fiber float64 xyz, track int32
// xyz) under `matrix`; `read(byteOffset, byteCount)` returns array bytes.
template<class Read>
double polylineLength(const Read& read,int64_t begin,int64_t end,const std::string& kind,const VolumeAffine& matrix) {
    const auto stride=kind=="track"?12:24;
    double length=0;std::optional<std::array<double,3>> previous;
    for(int64_t start=begin;start<end;start+=256) {
        const auto n=std::min<int64_t>(256,end-start);
        const std::vector<uint8_t> bytes=read(start*stride,n*stride);
        for(int64_t i=0;i<n;++i) {
            std::array<double,3> p;
            if(stride==24)copyScalars<double>(std::span(bytes).subspan(i*stride,stride),p);
            else {std::array<int32_t,3> q;copyScalars<int32_t>(std::span(bytes).subspan(i*stride,stride),q);for(int k=0;k<3;++k)p[k]=q[k];}
            // Transform displacements directly to avoid cancellation from translation.
            if(previous){auto delta=p;for(int k=0;k<3;++k)delta[k]-=(*previous)[k];delta=matrix.apply(delta,true);length+=std::hypot(delta[0],delta[1],delta[2]);}
            previous=p;
        }
    }
    return length;
}
// Rows of the length array of a stored revision (see lengthRows).
inline LengthRows lengthRows(GeometryReader& reader,const std::string& kind) {
    const auto& arrays=reader.manifest().at("arrays");
    return lengthRows(kind,[&](const std::string& name){return arrays.at(name).at("shape")[0].get<int64_t>();},
        [&](const std::string& name,int64_t offset,int64_t bytes){return reader.read(name,offset,bytes);});
}
inline double transformedLength(sqlite3* db,const std::string& revision,const VolumeAffine& matrix,const std::string& kind) {
    const auto key=revision+matrix.json().dump();
    auto pool=activeStatementPools.find(db);
    if(pool!=activeStatementPools.end()&&pool->second->readSnapshot) {
        const auto found=pool->second->transformedLengths.find(key);
        if(found!=pool->second->transformedLengths.end())return found->second;
    }
    GeometryReader reader(db,revision);
    const auto rows=lengthRows(reader,kind);
    const auto length=polylineLength([&](int64_t offset,int64_t bytes){return reader.read(rows.name,offset,bytes);},
        rows.begin,rows.end,kind,matrix);
    if(pool!=activeStatementPools.end()&&pool->second->readSnapshot) {
        auto& cache=pool->second->transformedLengths;
        if(cache.size()>=1024)cache.clear();
        cache.emplace(key,length);
    }
    return length;
}
}
