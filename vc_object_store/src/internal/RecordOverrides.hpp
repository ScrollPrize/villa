#pragma once
#include "internal/RecordResolver.hpp"
namespace vc::objects::detail {
// Rebase location hints and register extension schemas when metadata moves
// between owners. Payload references are retained, never copied.
void prepareOverrideReferences(sqlite3* source,sqlite3* destination,ObjectRecord&);
// Copy only the selected record/descriptor into the writable overlay. No
// payloads, aliases or history bases are materialized. Normal mutations then
// advance its inherited version once, using their ordinary validation path.
bool createRecordOverride(sqlite3*,const char* table,const std::string& uuid);
}
