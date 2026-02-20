#include "test.hpp"

#include "vc/core/Version.hpp"

TEST(ProjectInfo, NameIsNonEmpty)
{
    EXPECT_FALSE(ProjectInfo::Name().empty());
}

TEST(ProjectInfo, VersionComponentsAreConsistent)
{
    auto expected = std::to_string(ProjectInfo::VersionMajor()) + "." +
                    std::to_string(ProjectInfo::VersionMinor()) + "." +
                    std::to_string(ProjectInfo::VersionPatch());
    EXPECT_EQ(ProjectInfo::VersionString(), expected);
}

TEST(ProjectInfo, NameAndVersionContainsBoth)
{
    auto nav = ProjectInfo::NameAndVersion();
    EXPECT_NE(nav.find(ProjectInfo::Name()), std::string::npos);
    EXPECT_NE(nav.find(ProjectInfo::VersionString()), std::string::npos);
}

TEST(ProjectInfo, RepositoryHashIsNonEmpty)
{
    EXPECT_FALSE(ProjectInfo::RepositoryHash().empty());
    EXPECT_FALSE(ProjectInfo::RepositoryShortHash().empty());
}
