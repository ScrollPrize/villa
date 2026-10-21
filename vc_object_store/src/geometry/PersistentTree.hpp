#pragma once
#include "geometry/SharedGeometry.hpp"

namespace vc::objects::detail {
using Tree = nlohmann::json;
// Immutable AVL sequence. References carry authenticated subtree summaries;
// nodes contain two references or a leaf value. Empty trees are JSON null.
class TreeCacheScope {
    bool owner_;
public:
    TreeCacheScope();
    ~TreeCacheScope();
    TreeCacheScope(const TreeCacheScope&)=delete;
    TreeCacheScope& operator=(const TreeCacheScope&)=delete;
};
Tree treeNode(sqlite3*, const Tree&);
Tree treeLookup(sqlite3*, const Tree&, const std::string&);
Tree treeLeaf(sqlite3*, Tree value, int64_t size, const Tree& bounds = nullptr,
              bool directional = false, const std::string& key = {});
Tree treeJoin(sqlite3*, const Tree&, const Tree&);
Tree treeBuild(sqlite3*, const std::vector<Tree>&);
int64_t treeSize(const Tree&);
using LeafSlice = std::function<Tree(const Tree&, int64_t, int64_t)>;
Tree treeSlice(sqlite3*, const Tree&, int64_t start, int64_t count, const LeafSlice&);
void treeVisit(sqlite3*, const Tree&, int64_t start, int64_t count,
               const std::function<void(const Tree&, int64_t, int64_t)>&);
Tree treeGet(sqlite3*, const Tree&, const std::string& key);
Tree treePut(sqlite3*, const Tree&, const std::string& key, const Tree& leaf);
void treeEntries(sqlite3*, const Tree&, const std::function<void(const Tree&,const Tree&)>&);
void treeWalk(sqlite3*, const Tree&, const std::function<void(const Tree&)>&);
Tree mergeTreeBounds(const Tree&, const Tree&);
Tree coordinateBounds(std::span<const uint8_t>, const std::string&, bool patch);
Tree indexedDescriptor(sqlite3*, GeometryReader&, const std::string& name, bool coordinates, bool patch=false,
                       const std::array<int64_t,2>& origin={0,0});
Tree indexedSurfaceArray(sqlite3*, const GeometryArray&, int64_t tile, const std::array<int64_t,2>& origin);
Tree byteTree(sqlite3*, const std::string&, const std::string&, std::span<const uint8_t>, bool coordinates);
Tree byteSlice(sqlite3*, const Tree&, const std::string&, const std::string&, int64_t, int64_t, bool coordinates);
Tree byteJoin(sqlite3*, const Tree&, const Tree&, const std::string&, const std::string&, bool coordinates);
std::vector<uint8_t> treeBytes(sqlite3*, const Tree&, const std::string&, int64_t, int64_t);
std::string tileKey(int64_t row, int64_t column);
std::string identityKey(int64_t id);
Tree packAnnotation(sqlite3*, const Tree&);
Tree unpackAnnotation(sqlite3*, const Tree&);
bool annotationDirectional(const Tree&);
Tree identityGet(sqlite3*, const Tree&, int64_t);
Tree unionIdentityLedgers(sqlite3*, const Tree&, const Tree&);
Tree identityPut(sqlite3*, const Tree&, int64_t, const Tree&);
Tree tileHalos(sqlite3*, const Tree&, int64_t tile, const std::set<std::pair<int64_t,int64_t>>&);
void transferTree(sqlite3* source, sqlite3* destination, const Tree&);
}
