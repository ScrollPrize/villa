#pragma once

#include <QColor>
#include <QString>
#include <Qt>

#include <array>
#include <cstddef>
#include <optional>

namespace vc3d::spiral {

// The point-collection roles the Spiral workspace draws, edits, and commits.
// Both share one PointCollections v1 document shape and one client workflow;
// the fitter tells them apart only by the file they live in and by `wind_a`:
//   - same-winding: no `wind_a`; every point sits on one winding.
//   - relative-winding: an integer `wind_a` per point whose pairwise
//     differences are the winding constraint (the values themselves are not
//     absolute). New collections count 0, 1, 2, ... in placement order.
enum class PclRole { SameWinding = 0, Relative = 1 };

constexpr std::array<PclRole, 2> kEditablePclRoles{
    PclRole::SameWinding, PclRole::Relative};

constexpr std::size_t pclRoleIndex(PclRole role)
{
    return static_cast<std::size_t>(role);
}

// Whether points of this role carry a winding annotation.
constexpr bool pclRoleHasWindingAnnotations(PclRole role)
{
    return role == PclRole::Relative;
}

// Keyboard toggle for the role's point-placement mode in the Spiral viewers.
constexpr Qt::Key pclRoleToggleKey(PclRole role)
{
    return role == PclRole::Relative ? Qt::Key_E : Qt::Key_Q;
}

// The service's `role` value for uploads and status entries.
inline QString pclRoleName(PclRole role)
{
    return role == PclRole::Relative ? QStringLiteral("relative")
                                     : QStringLiteral("same_winding");
}

inline std::optional<PclRole> pclRoleFromName(const QString& name)
{
    for (const PclRole role : kEditablePclRoles) {
        if (pclRoleName(role) == name) return role;
    }
    return std::nullopt;
}

// Conventional dataset file the service commits this role into.
inline QString pclRoleFileName(PclRole role)
{
    return role == PclRole::Relative
        ? QStringLiteral("relative_windings.json")
        : QStringLiteral("same_windings.json");
}

// /session/status key carrying the role's display artifact reference.
inline QString pclRoleStatusKey(PclRole role)
{
    return role == PclRole::Relative
        ? QStringLiteral("relative_winding_artifact")
        : QStringLiteral("same_winding_artifact");
}

// Human label for messages ("same-winding collection 3").
inline QString pclRoleDisplayName(PclRole role)
{
    return role == PclRole::Relative ? QStringLiteral("relative-winding")
                                     : QStringLiteral("same-winding");
}

// Prefix of generated collection names and upload ids.
inline QString pclRoleCollectionPrefix(PclRole role)
{
    return role == PclRole::Relative ? QStringLiteral("relative_winding")
                                     : QStringLiteral("same_winding");
}

// Accent color of the role's placement cursor and default collection color.
inline QColor pclRoleAccentColor(PclRole role)
{
    return role == PclRole::Relative ? QColor(255, 170, 50)
                                     : QColor(50, 255, 215);
}

} // namespace vc3d::spiral
