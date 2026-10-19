#pragma once
#include <sqlite3.h>
#include <string>
namespace vc::objects::detail {
// Copy retained immutable history, comparing identities and verifying chunk hashes.
void transferGeometry(sqlite3* source, sqlite3* destination, const std::string& objectUuid);
}
