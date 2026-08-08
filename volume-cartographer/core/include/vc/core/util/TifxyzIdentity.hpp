#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace vc::util {

/** Resolve the UUID written to a TIFXYZ meta.json.
 *
 * An explicit UUID lets production pipelines keep artifact directories named
 * after a processing role (for example, "output_tifxyz") without giving
 * unrelated surfaces the same collection identity. With no override, retain
 * the historical output-directory-basename behavior.
 */
inline std::string resolveTifxyzUuid(
    const std::filesystem::path& outputDirectory,
    const std::filesystem::path& inputPath,
    const std::string& explicitUuid = {})
{
    if (!explicitUuid.empty()) {
        return explicitUuid;
    }

    std::string uuid = outputDirectory.filename().string();
    if (uuid.empty()) {
        uuid = inputPath.stem().string();
    }
    if (uuid.empty()) {
        throw std::invalid_argument(
            "cannot derive TIFXYZ UUID from an empty output directory and input path");
    }
    return uuid;
}

}  // namespace vc::util
