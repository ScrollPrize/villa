#pragma once
#include <sqlite3.h>
#include <string>
#include <cstdint>
namespace vc::objects::detail {
// Copy retained immutable history, comparing identities and verifying chunk hashes.
void transferGeometry(sqlite3* source, sqlite3* destination, const std::string& objectUuid, int64_t afterRow = 0);
}
