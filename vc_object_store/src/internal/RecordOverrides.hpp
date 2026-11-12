#pragma once
#include "internal/RecordResolver.hpp"
namespace vc::objects::detail {
// Location changes preserve the target, not the spelling of a relative hint.
std::string rebaseExternalLocator(const std::string&, const std::filesystem::path& source,
                                 const std::filesystem::path& destination);
// Private export/conversion snapshot only, inside the caller's transaction.
// Rewrites current and retained record hints without publishing logical edits.
void relocateExternalLocators(sqlite3*, const std::filesystem::path& source,
                              const std::filesystem::path& destination);
// Rebase location hints and register extension schemas when metadata moves
// between owners. Payload references are retained, never copied.
void prepareOverrideReferences(sqlite3* source,sqlite3* destination,ObjectRecord&);
// Copy only the selected record/descriptor into the writable overlay. No
// payloads, aliases or history bases are materialized. Normal mutations then
// advance its inherited version once, using their ordinary validation path.
bool createRecordOverride(sqlite3*,const char* table,const std::string& uuid);
}
