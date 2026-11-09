#pragma once
#include <filesystem>
#include <memory>
struct sqlite3;
namespace vc::objects::detail {
// Offline maintenance already holds the root connection exclusively. Reuse
// its snapshot, open only sealed dependencies, and install the common verified
// reference resolver until this guard is destroyed.
std::shared_ptr<void> attachCollectionReferenceResolver(sqlite3*, const std::filesystem::path&);
}
