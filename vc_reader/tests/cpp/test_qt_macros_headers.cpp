// Compile smoke test guarding public vc_reader headers against Qt keyword macros
// (built with -Dslots= -Dsignals=public -Demit=, as in Qt6 targets without QT_NO_KEYWORDS).
#include <vc_reader/array.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/reader.hpp>
#include <vc_reader/source_reader.hpp>
#include <vc_reader/native/zarr_storage.hpp>
#include <vc_reader/native/shard_index.hpp>
#include <vc_reader/native/channel_policy.hpp>

int main()
{
    return 0;
}
