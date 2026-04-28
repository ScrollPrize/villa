#include "test.hpp"

#include "vc/core/cache/BlockPipeline.hpp"

#include <cstdint>
#include <vector>

using vc::cache::BlockPipeline;
using V = BlockPipeline::FetchVerdict;

static V classify(const std::vector<uint8_t>& bytes,
                  bool threw,
                  bool sourceConfirmsAbsent,
                  bool transient,
                  bool authProven)
{
    return BlockPipeline::classifyFetch(bytes, threw, sourceConfirmsAbsent, transient, authProven);
}

static std::vector<uint8_t> data(size_t n, uint8_t v = 0x42)
{
    return std::vector<uint8_t>(n, v);
}

static std::vector<uint8_t> zeros(size_t n)
{
    return std::vector<uint8_t>(n, 0);
}

// The user's contract:
// "is the chunk on the server? yes? then its not negative cached. is it not on
//  the server? then it is negative cached. did we determine that it truly isnt
//  on the server? thats literally the only time we negative cache. transient
//  error? no negative caching. aws issues? no negative cache"
//
// Mapping: only DefinitivelyAbsent ever causes negative caching. So
// DefinitivelyAbsent must require BOTH a verified source-side absence
// (sourceConfirmsAbsent) AND a session-wide proof of auth (authProven). Any
// other input shape must be HasData or RetryLater.

TEST(classifyFetch, has_data_when_bytes_nonempty_and_no_errors)
{
    EXPECT_EQ(classify(data(64), false, false, false, true), V::HasData);
    EXPECT_EQ(classify(data(64), false, false, false, false), V::HasData);
}

TEST(classifyFetch, all_zero_bytes_are_real_data_not_absent)
{
    // For zarr v2 raw uncompressed chunks a region that legitimately
    // contains zeros is served as all-zero bytes. Treating that as
    // "absent" poisons the cache and is wrong.
    EXPECT_NE(classify(zeros(64), false, false, false, true), V::EmptyConfirmed);
    EXPECT_NE(classify(zeros(64), false, false, false, false), V::EmptyConfirmed);
    EXPECT_EQ(classify(zeros(64), false, false, false, true), V::HasData);
}

TEST(classifyFetch, threw_is_retry_later_never_absent)
{
    EXPECT_EQ(classify({}, true, false, false, true), V::Transient);
    EXPECT_EQ(classify({}, true, true, false, true), V::Transient);
    EXPECT_EQ(classify({}, true, true, true, true), V::Transient);
    EXPECT_EQ(classify(data(64), true, true, true, true), V::Transient);
}

TEST(classifyFetch, transient_is_retry_later_never_absent)
{
    EXPECT_EQ(classify({}, false, false, true, true), V::Transient);
    EXPECT_EQ(classify({}, false, true, true, true), V::Transient);
    EXPECT_EQ(classify({}, false, true, true, false), V::Transient);
}

TEST(classifyFetch, empty_bytes_with_no_confirmation_is_retry_later)
{
    EXPECT_EQ(classify({}, false, false, false, true), V::Transient);
    EXPECT_EQ(classify({}, false, false, false, false), V::Transient);
}

TEST(classifyFetch, definitively_absent_only_when_verified_and_auth_proven)
{
    EXPECT_EQ(classify({}, false, true, false, true), V::EmptyConfirmed);
}

TEST(classifyFetch, sourceConfirmsAbsent_without_authProven_is_retry_later)
{
    // We saw a 404 but we never proved auth was good — that 404 might just
    // be the bucket lying because we are unauthorized.
    EXPECT_EQ(classify({}, false, true, false, false), V::Transient);
}

TEST(classifyFetch, threw_overrides_everything)
{
    EXPECT_EQ(classify({}, true, true, false, true), V::Transient);
}

TEST(classifyFetch, transient_overrides_sourceConfirmsAbsent)
{
    EXPECT_EQ(classify({}, false, true, true, true), V::Transient);
}
