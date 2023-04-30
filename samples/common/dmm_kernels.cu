#include "dmm_generator_private.h"
#include "optix_micromap.h"

using namespace shared;

static constexpr uint32_t WarpSize = 32;
