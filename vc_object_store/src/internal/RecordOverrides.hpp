#pragma once
#include "internal/RecordResolver.hpp"
namespace vc::objects::detail {
// Copy only the selected record/descriptor into the writable overlay. No
// payloads, aliases or history bases are materialized. Normal mutations then
// advance its inherited version once, using their ordinary validation path.
bool createRecordOverride(sqlite3*,const char* table,const std::string& uuid);
}
