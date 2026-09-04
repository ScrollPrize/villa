#include "vc/core/util/OpenMPConfig.hpp"

#include <cstdlib>
#include <cstring>

#include <omp.h>

namespace vc::core::util
{

void disableOpenMPDynamicTeams()
{
    // Opt-out, for bisecting a regression against the previous behaviour.
    if (const char* keep = std::getenv("VC_OMP_DYNAMIC");
        keep && std::strcmp(keep, "0") != 0 && keep[0] != '\0') {
        return;
    }
    omp_set_dynamic(0);
}

}  // namespace vc::core::util
