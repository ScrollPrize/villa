#pragma once

#include "vc/core/util/CostFunctions.hpp"

#define OPTIMIZE_ALL 1
#define SURF_LOSS 2
#define SPACE_LOSS 2 //SURF and SPACE are never used together
#define LOSS_3D_INDIRECT 4
#define LOSS_ZLOC 8
#define FLAG_GEN0 16
#define LOSS_ON_SURF 32
#define LOSS_ON_NORMALS 64

#define STATE_UNUSED 0
#define STATE_LOC_VALID 1
#define STATE_PROCESSING 2
#define STATE_COORD_VALID 4
#define STATE_FAIL 8
#define STATE_PHYS_ONLY 16
