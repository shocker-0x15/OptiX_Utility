#pragma once

#include "micro_map_generator_private.h"
#include "dmm_generator.h"

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
CUDA_CONSTANT_MEM constexpr float2 microVertexBarycentrics[] = {
    // Level 0
    float2{ 0.000000f, 0.000000f }, float2{ 1.000000f, 0.000000f }, float2{ 0.000000f, 1.000000f },
    // + Level 1
    float2{ 0.000000f, 0.500000f }, float2{ 0.500000f, 0.500000f }, float2{ 0.500000f, 0.000000f },
    // + Level 2
    float2{ 0.000000f, 0.250000f }, float2{ 0.250000f, 0.250000f }, float2{ 0.250000f, 0.000000f },
    float2{ 0.500000f, 0.250000f }, float2{ 0.750000f, 0.250000f }, float2{ 0.750000f, 0.000000f },
    float2{ 0.000000f, 0.750000f }, float2{ 0.250000f, 0.750000f }, float2{ 0.250000f, 0.500000f },
    // + Level 3
    float2{ 0.000000f, 0.125000f }, float2{ 0.125000f, 0.125000f }, float2{ 0.125000f, 0.000000f },
    float2{ 0.250000f, 0.125000f }, float2{ 0.375000f, 0.125000f }, float2{ 0.375000f, 0.000000f },
    float2{ 0.000000f, 0.375000f }, float2{ 0.125000f, 0.375000f }, float2{ 0.125000f, 0.250000f },
    float2{ 0.250000f, 0.375000f }, float2{ 0.375000f, 0.375000f }, float2{ 0.375000f, 0.250000f },
    float2{ 0.500000f, 0.125000f }, float2{ 0.625000f, 0.125000f }, float2{ 0.625000f, 0.000000f },
    float2{ 0.750000f, 0.125000f }, float2{ 0.875000f, 0.125000f }, float2{ 0.875000f, 0.000000f },
    float2{ 0.500000f, 0.375000f }, float2{ 0.625000f, 0.375000f }, float2{ 0.625000f, 0.250000f },
    float2{ 0.250000f, 0.625000f }, float2{ 0.375000f, 0.625000f }, float2{ 0.375000f, 0.500000f },
    float2{ 0.000000f, 0.625000f }, float2{ 0.125000f, 0.625000f }, float2{ 0.125000f, 0.500000f },
    float2{ 0.000000f, 0.875000f }, float2{ 0.125000f, 0.875000f }, float2{ 0.125000f, 0.750000f },
    // + Level 4
    float2{ 0.000000f, 0.062500f }, float2{ 0.062500f, 0.062500f }, float2{ 0.062500f, 0.000000f },
    float2{ 0.125000f, 0.062500f }, float2{ 0.187500f, 0.062500f }, float2{ 0.187500f, 0.000000f },
    float2{ 0.000000f, 0.187500f }, float2{ 0.062500f, 0.187500f }, float2{ 0.062500f, 0.125000f },
    float2{ 0.125000f, 0.187500f }, float2{ 0.187500f, 0.187500f }, float2{ 0.187500f, 0.125000f },
    float2{ 0.250000f, 0.062500f }, float2{ 0.312500f, 0.062500f }, float2{ 0.312500f, 0.000000f },
    float2{ 0.375000f, 0.062500f }, float2{ 0.437500f, 0.062500f }, float2{ 0.437500f, 0.000000f },
    float2{ 0.250000f, 0.187500f }, float2{ 0.312500f, 0.187500f }, float2{ 0.312500f, 0.125000f },
    float2{ 0.125000f, 0.312500f }, float2{ 0.187500f, 0.312500f }, float2{ 0.187500f, 0.250000f },
    float2{ 0.000000f, 0.312500f }, float2{ 0.062500f, 0.312500f }, float2{ 0.062500f, 0.250000f },
    float2{ 0.000000f, 0.437500f }, float2{ 0.062500f, 0.437500f }, float2{ 0.062500f, 0.375000f },
    float2{ 0.125000f, 0.437500f }, float2{ 0.187500f, 0.437500f }, float2{ 0.187500f, 0.375000f },
    float2{ 0.250000f, 0.312500f }, float2{ 0.312500f, 0.312500f }, float2{ 0.312500f, 0.250000f },
    float2{ 0.375000f, 0.312500f }, float2{ 0.437500f, 0.312500f }, float2{ 0.437500f, 0.250000f },
    float2{ 0.250000f, 0.437500f }, float2{ 0.312500f, 0.437500f }, float2{ 0.312500f, 0.375000f },
    float2{ 0.375000f, 0.437500f }, float2{ 0.437500f, 0.437500f }, float2{ 0.437500f, 0.375000f },
    float2{ 0.375000f, 0.187500f }, float2{ 0.437500f, 0.187500f }, float2{ 0.437500f, 0.125000f },
    float2{ 0.500000f, 0.062500f }, float2{ 0.562500f, 0.062500f }, float2{ 0.562500f, 0.000000f },
    float2{ 0.625000f, 0.062500f }, float2{ 0.687500f, 0.062500f }, float2{ 0.687500f, 0.000000f },
    float2{ 0.500000f, 0.187500f }, float2{ 0.562500f, 0.187500f }, float2{ 0.562500f, 0.125000f },
    float2{ 0.625000f, 0.187500f }, float2{ 0.687500f, 0.187500f }, float2{ 0.687500f, 0.125000f },
    float2{ 0.750000f, 0.062500f }, float2{ 0.812500f, 0.062500f }, float2{ 0.812500f, 0.000000f },
    float2{ 0.875000f, 0.062500f }, float2{ 0.937500f, 0.062500f }, float2{ 0.937500f, 0.000000f },
    float2{ 0.750000f, 0.187500f }, float2{ 0.812500f, 0.187500f }, float2{ 0.812500f, 0.125000f },
    float2{ 0.625000f, 0.312500f }, float2{ 0.687500f, 0.312500f }, float2{ 0.687500f, 0.250000f },
    float2{ 0.500000f, 0.312500f }, float2{ 0.562500f, 0.312500f }, float2{ 0.562500f, 0.250000f },
    float2{ 0.500000f, 0.437500f }, float2{ 0.562500f, 0.437500f }, float2{ 0.562500f, 0.375000f },
    float2{ 0.375000f, 0.562500f }, float2{ 0.437500f, 0.562500f }, float2{ 0.437500f, 0.500000f },
    float2{ 0.250000f, 0.562500f }, float2{ 0.312500f, 0.562500f }, float2{ 0.312500f, 0.500000f },
    float2{ 0.250000f, 0.687500f }, float2{ 0.312500f, 0.687500f }, float2{ 0.312500f, 0.625000f },
    float2{ 0.125000f, 0.687500f }, float2{ 0.187500f, 0.687500f }, float2{ 0.187500f, 0.625000f },
    float2{ 0.125000f, 0.562500f }, float2{ 0.187500f, 0.562500f }, float2{ 0.187500f, 0.500000f },
    float2{ 0.000000f, 0.562500f }, float2{ 0.062500f, 0.562500f }, float2{ 0.062500f, 0.500000f },
    float2{ 0.000000f, 0.687500f }, float2{ 0.062500f, 0.687500f }, float2{ 0.062500f, 0.625000f },
    float2{ 0.000000f, 0.812500f }, float2{ 0.062500f, 0.812500f }, float2{ 0.062500f, 0.750000f },
    float2{ 0.125000f, 0.812500f }, float2{ 0.187500f, 0.812500f }, float2{ 0.187500f, 0.750000f },
    float2{ 0.000000f, 0.937500f }, float2{ 0.062500f, 0.937500f }, float2{ 0.062500f, 0.875000f },
    // + Level 5
    float2{ 0.000000f, 0.031250f }, float2{ 0.031250f, 0.031250f }, float2{ 0.031250f, 0.000000f },
    float2{ 0.062500f, 0.031250f }, float2{ 0.093750f, 0.031250f }, float2{ 0.093750f, 0.000000f },
    float2{ 0.000000f, 0.093750f }, float2{ 0.031250f, 0.093750f }, float2{ 0.031250f, 0.062500f },
    float2{ 0.062500f, 0.093750f }, float2{ 0.093750f, 0.093750f }, float2{ 0.093750f, 0.062500f },
    float2{ 0.125000f, 0.031250f }, float2{ 0.156250f, 0.031250f }, float2{ 0.156250f, 0.000000f },
    float2{ 0.187500f, 0.031250f }, float2{ 0.218750f, 0.031250f }, float2{ 0.218750f, 0.000000f },
    float2{ 0.125000f, 0.093750f }, float2{ 0.156250f, 0.093750f }, float2{ 0.156250f, 0.062500f },
    float2{ 0.062500f, 0.156250f }, float2{ 0.093750f, 0.156250f }, float2{ 0.093750f, 0.125000f },
    float2{ 0.000000f, 0.156250f }, float2{ 0.031250f, 0.156250f }, float2{ 0.031250f, 0.125000f },
    float2{ 0.000000f, 0.218750f }, float2{ 0.031250f, 0.218750f }, float2{ 0.031250f, 0.187500f },
    float2{ 0.062500f, 0.218750f }, float2{ 0.093750f, 0.218750f }, float2{ 0.093750f, 0.187500f },
    float2{ 0.125000f, 0.156250f }, float2{ 0.156250f, 0.156250f }, float2{ 0.156250f, 0.125000f },
    float2{ 0.187500f, 0.156250f }, float2{ 0.218750f, 0.156250f }, float2{ 0.218750f, 0.125000f },
    float2{ 0.125000f, 0.218750f }, float2{ 0.156250f, 0.218750f }, float2{ 0.156250f, 0.187500f },
    float2{ 0.187500f, 0.218750f }, float2{ 0.218750f, 0.218750f }, float2{ 0.218750f, 0.187500f },
    float2{ 0.187500f, 0.093750f }, float2{ 0.218750f, 0.093750f }, float2{ 0.218750f, 0.062500f },
    float2{ 0.250000f, 0.031250f }, float2{ 0.281250f, 0.031250f }, float2{ 0.281250f, 0.000000f },
    float2{ 0.312500f, 0.031250f }, float2{ 0.343750f, 0.031250f }, float2{ 0.343750f, 0.000000f },
    float2{ 0.250000f, 0.093750f }, float2{ 0.281250f, 0.093750f }, float2{ 0.281250f, 0.062500f },
    float2{ 0.312500f, 0.093750f }, float2{ 0.343750f, 0.093750f }, float2{ 0.343750f, 0.062500f },
    float2{ 0.375000f, 0.031250f }, float2{ 0.406250f, 0.031250f }, float2{ 0.406250f, 0.000000f },
    float2{ 0.437500f, 0.031250f }, float2{ 0.468750f, 0.031250f }, float2{ 0.468750f, 0.000000f },
    float2{ 0.375000f, 0.093750f }, float2{ 0.406250f, 0.093750f }, float2{ 0.406250f, 0.062500f },
    float2{ 0.312500f, 0.156250f }, float2{ 0.343750f, 0.156250f }, float2{ 0.343750f, 0.125000f },
    float2{ 0.250000f, 0.156250f }, float2{ 0.281250f, 0.156250f }, float2{ 0.281250f, 0.125000f },
    float2{ 0.250000f, 0.218750f }, float2{ 0.281250f, 0.218750f }, float2{ 0.281250f, 0.187500f },
    float2{ 0.187500f, 0.281250f }, float2{ 0.218750f, 0.281250f }, float2{ 0.218750f, 0.250000f },
    float2{ 0.125000f, 0.281250f }, float2{ 0.156250f, 0.281250f }, float2{ 0.156250f, 0.250000f },
    float2{ 0.125000f, 0.343750f }, float2{ 0.156250f, 0.343750f }, float2{ 0.156250f, 0.312500f },
    float2{ 0.062500f, 0.343750f }, float2{ 0.093750f, 0.343750f }, float2{ 0.093750f, 0.312500f },
    float2{ 0.062500f, 0.281250f }, float2{ 0.093750f, 0.281250f }, float2{ 0.093750f, 0.250000f },
    float2{ 0.000000f, 0.281250f }, float2{ 0.031250f, 0.281250f }, float2{ 0.031250f, 0.250000f },
    float2{ 0.000000f, 0.343750f }, float2{ 0.031250f, 0.343750f }, float2{ 0.031250f, 0.312500f },
    float2{ 0.000000f, 0.406250f }, float2{ 0.031250f, 0.406250f }, float2{ 0.031250f, 0.375000f },
    float2{ 0.062500f, 0.406250f }, float2{ 0.093750f, 0.406250f }, float2{ 0.093750f, 0.375000f },
    float2{ 0.000000f, 0.468750f }, float2{ 0.031250f, 0.468750f }, float2{ 0.031250f, 0.437500f },
    float2{ 0.062500f, 0.468750f }, float2{ 0.093750f, 0.468750f }, float2{ 0.093750f, 0.437500f },
    float2{ 0.125000f, 0.406250f }, float2{ 0.156250f, 0.406250f }, float2{ 0.156250f, 0.375000f },
    float2{ 0.187500f, 0.406250f }, float2{ 0.218750f, 0.406250f }, float2{ 0.218750f, 0.375000f },
    float2{ 0.125000f, 0.468750f }, float2{ 0.156250f, 0.468750f }, float2{ 0.156250f, 0.437500f },
    float2{ 0.187500f, 0.468750f }, float2{ 0.218750f, 0.468750f }, float2{ 0.218750f, 0.437500f },
    float2{ 0.187500f, 0.343750f }, float2{ 0.218750f, 0.343750f }, float2{ 0.218750f, 0.312500f },
    float2{ 0.250000f, 0.281250f }, float2{ 0.281250f, 0.281250f }, float2{ 0.281250f, 0.250000f },
    float2{ 0.312500f, 0.281250f }, float2{ 0.343750f, 0.281250f }, float2{ 0.343750f, 0.250000f },
    float2{ 0.250000f, 0.343750f }, float2{ 0.281250f, 0.343750f }, float2{ 0.281250f, 0.312500f },
    float2{ 0.312500f, 0.343750f }, float2{ 0.343750f, 0.343750f }, float2{ 0.343750f, 0.312500f },
    float2{ 0.375000f, 0.281250f }, float2{ 0.406250f, 0.281250f }, float2{ 0.406250f, 0.250000f },
    float2{ 0.437500f, 0.281250f }, float2{ 0.468750f, 0.281250f }, float2{ 0.468750f, 0.250000f },
    float2{ 0.375000f, 0.343750f }, float2{ 0.406250f, 0.343750f }, float2{ 0.406250f, 0.312500f },
    float2{ 0.312500f, 0.406250f }, float2{ 0.343750f, 0.406250f }, float2{ 0.343750f, 0.375000f },
    float2{ 0.250000f, 0.406250f }, float2{ 0.281250f, 0.406250f }, float2{ 0.281250f, 0.375000f },
    float2{ 0.250000f, 0.468750f }, float2{ 0.281250f, 0.468750f }, float2{ 0.281250f, 0.437500f },
    float2{ 0.312500f, 0.468750f }, float2{ 0.343750f, 0.468750f }, float2{ 0.343750f, 0.437500f },
    float2{ 0.375000f, 0.406250f }, float2{ 0.406250f, 0.406250f }, float2{ 0.406250f, 0.375000f },
    float2{ 0.437500f, 0.406250f }, float2{ 0.468750f, 0.406250f }, float2{ 0.468750f, 0.375000f },
    float2{ 0.375000f, 0.468750f }, float2{ 0.406250f, 0.468750f }, float2{ 0.406250f, 0.437500f },
    float2{ 0.437500f, 0.468750f }, float2{ 0.468750f, 0.468750f }, float2{ 0.468750f, 0.437500f },
    float2{ 0.437500f, 0.343750f }, float2{ 0.468750f, 0.343750f }, float2{ 0.468750f, 0.312500f },
    float2{ 0.437500f, 0.218750f }, float2{ 0.468750f, 0.218750f }, float2{ 0.468750f, 0.187500f },
    float2{ 0.437500f, 0.156250f }, float2{ 0.468750f, 0.156250f }, float2{ 0.468750f, 0.125000f },
    float2{ 0.375000f, 0.156250f }, float2{ 0.406250f, 0.156250f }, float2{ 0.406250f, 0.125000f },
    float2{ 0.375000f, 0.218750f }, float2{ 0.406250f, 0.218750f }, float2{ 0.406250f, 0.187500f },
    float2{ 0.312500f, 0.218750f }, float2{ 0.343750f, 0.218750f }, float2{ 0.343750f, 0.187500f },
    float2{ 0.437500f, 0.093750f }, float2{ 0.468750f, 0.093750f }, float2{ 0.468750f, 0.062500f },
    float2{ 0.500000f, 0.031250f }, float2{ 0.531250f, 0.031250f }, float2{ 0.531250f, 0.000000f },
    float2{ 0.562500f, 0.031250f }, float2{ 0.593750f, 0.031250f }, float2{ 0.593750f, 0.000000f },
    float2{ 0.500000f, 0.093750f }, float2{ 0.531250f, 0.093750f }, float2{ 0.531250f, 0.062500f },
    float2{ 0.562500f, 0.093750f }, float2{ 0.593750f, 0.093750f }, float2{ 0.593750f, 0.062500f },
    float2{ 0.625000f, 0.031250f }, float2{ 0.656250f, 0.031250f }, float2{ 0.656250f, 0.000000f },
    float2{ 0.687500f, 0.031250f }, float2{ 0.718750f, 0.031250f }, float2{ 0.718750f, 0.000000f },
    float2{ 0.625000f, 0.093750f }, float2{ 0.656250f, 0.093750f }, float2{ 0.656250f, 0.062500f },
    float2{ 0.562500f, 0.156250f }, float2{ 0.593750f, 0.156250f }, float2{ 0.593750f, 0.125000f },
    float2{ 0.500000f, 0.156250f }, float2{ 0.531250f, 0.156250f }, float2{ 0.531250f, 0.125000f },
    float2{ 0.500000f, 0.218750f }, float2{ 0.531250f, 0.218750f }, float2{ 0.531250f, 0.187500f },
    float2{ 0.562500f, 0.218750f }, float2{ 0.593750f, 0.218750f }, float2{ 0.593750f, 0.187500f },
    float2{ 0.625000f, 0.156250f }, float2{ 0.656250f, 0.156250f }, float2{ 0.656250f, 0.125000f },
    float2{ 0.687500f, 0.156250f }, float2{ 0.718750f, 0.156250f }, float2{ 0.718750f, 0.125000f },
    float2{ 0.625000f, 0.218750f }, float2{ 0.656250f, 0.218750f }, float2{ 0.656250f, 0.187500f },
    float2{ 0.687500f, 0.218750f }, float2{ 0.718750f, 0.218750f }, float2{ 0.718750f, 0.187500f },
    float2{ 0.687500f, 0.093750f }, float2{ 0.718750f, 0.093750f }, float2{ 0.718750f, 0.062500f },
    float2{ 0.750000f, 0.031250f }, float2{ 0.781250f, 0.031250f }, float2{ 0.781250f, 0.000000f },
    float2{ 0.812500f, 0.031250f }, float2{ 0.843750f, 0.031250f }, float2{ 0.843750f, 0.000000f },
    float2{ 0.750000f, 0.093750f }, float2{ 0.781250f, 0.093750f }, float2{ 0.781250f, 0.062500f },
    float2{ 0.812500f, 0.093750f }, float2{ 0.843750f, 0.093750f }, float2{ 0.843750f, 0.062500f },
    float2{ 0.875000f, 0.031250f }, float2{ 0.906250f, 0.031250f }, float2{ 0.906250f, 0.000000f },
    float2{ 0.937500f, 0.031250f }, float2{ 0.968750f, 0.031250f }, float2{ 0.968750f, 0.000000f },
    float2{ 0.875000f, 0.093750f }, float2{ 0.906250f, 0.093750f }, float2{ 0.906250f, 0.062500f },
    float2{ 0.812500f, 0.156250f }, float2{ 0.843750f, 0.156250f }, float2{ 0.843750f, 0.125000f },
    float2{ 0.750000f, 0.156250f }, float2{ 0.781250f, 0.156250f }, float2{ 0.781250f, 0.125000f },
    float2{ 0.750000f, 0.218750f }, float2{ 0.781250f, 0.218750f }, float2{ 0.781250f, 0.187500f },
    float2{ 0.687500f, 0.281250f }, float2{ 0.718750f, 0.281250f }, float2{ 0.718750f, 0.250000f },
    float2{ 0.625000f, 0.281250f }, float2{ 0.656250f, 0.281250f }, float2{ 0.656250f, 0.250000f },
    float2{ 0.625000f, 0.343750f }, float2{ 0.656250f, 0.343750f }, float2{ 0.656250f, 0.312500f },
    float2{ 0.562500f, 0.343750f }, float2{ 0.593750f, 0.343750f }, float2{ 0.593750f, 0.312500f },
    float2{ 0.562500f, 0.281250f }, float2{ 0.593750f, 0.281250f }, float2{ 0.593750f, 0.250000f },
    float2{ 0.500000f, 0.281250f }, float2{ 0.531250f, 0.281250f }, float2{ 0.531250f, 0.250000f },
    float2{ 0.500000f, 0.343750f }, float2{ 0.531250f, 0.343750f }, float2{ 0.531250f, 0.312500f },
    float2{ 0.500000f, 0.406250f }, float2{ 0.531250f, 0.406250f }, float2{ 0.531250f, 0.375000f },
    float2{ 0.562500f, 0.406250f }, float2{ 0.593750f, 0.406250f }, float2{ 0.593750f, 0.375000f },
    float2{ 0.500000f, 0.468750f }, float2{ 0.531250f, 0.468750f }, float2{ 0.531250f, 0.437500f },
    float2{ 0.437500f, 0.531250f }, float2{ 0.468750f, 0.531250f }, float2{ 0.468750f, 0.500000f },
    float2{ 0.375000f, 0.531250f }, float2{ 0.406250f, 0.531250f }, float2{ 0.406250f, 0.500000f },
    float2{ 0.375000f, 0.593750f }, float2{ 0.406250f, 0.593750f }, float2{ 0.406250f, 0.562500f },
    float2{ 0.312500f, 0.593750f }, float2{ 0.343750f, 0.593750f }, float2{ 0.343750f, 0.562500f },
    float2{ 0.312500f, 0.531250f }, float2{ 0.343750f, 0.531250f }, float2{ 0.343750f, 0.500000f },
    float2{ 0.250000f, 0.531250f }, float2{ 0.281250f, 0.531250f }, float2{ 0.281250f, 0.500000f },
    float2{ 0.250000f, 0.593750f }, float2{ 0.281250f, 0.593750f }, float2{ 0.281250f, 0.562500f },
    float2{ 0.250000f, 0.656250f }, float2{ 0.281250f, 0.656250f }, float2{ 0.281250f, 0.625000f },
    float2{ 0.312500f, 0.656250f }, float2{ 0.343750f, 0.656250f }, float2{ 0.343750f, 0.625000f },
    float2{ 0.250000f, 0.718750f }, float2{ 0.281250f, 0.718750f }, float2{ 0.281250f, 0.687500f },
    float2{ 0.187500f, 0.718750f }, float2{ 0.218750f, 0.718750f }, float2{ 0.218750f, 0.687500f },
    float2{ 0.187500f, 0.656250f }, float2{ 0.218750f, 0.656250f }, float2{ 0.218750f, 0.625000f },
    float2{ 0.125000f, 0.656250f }, float2{ 0.156250f, 0.656250f }, float2{ 0.156250f, 0.625000f },
    float2{ 0.125000f, 0.718750f }, float2{ 0.156250f, 0.718750f }, float2{ 0.156250f, 0.687500f },
    float2{ 0.062500f, 0.718750f }, float2{ 0.093750f, 0.718750f }, float2{ 0.093750f, 0.687500f },
    float2{ 0.187500f, 0.593750f }, float2{ 0.218750f, 0.593750f }, float2{ 0.218750f, 0.562500f },
    float2{ 0.187500f, 0.531250f }, float2{ 0.218750f, 0.531250f }, float2{ 0.218750f, 0.500000f },
    float2{ 0.125000f, 0.531250f }, float2{ 0.156250f, 0.531250f }, float2{ 0.156250f, 0.500000f },
    float2{ 0.125000f, 0.593750f }, float2{ 0.156250f, 0.593750f }, float2{ 0.156250f, 0.562500f },
    float2{ 0.062500f, 0.593750f }, float2{ 0.093750f, 0.593750f }, float2{ 0.093750f, 0.562500f },
    float2{ 0.062500f, 0.531250f }, float2{ 0.093750f, 0.531250f }, float2{ 0.093750f, 0.500000f },
    float2{ 0.000000f, 0.531250f }, float2{ 0.031250f, 0.531250f }, float2{ 0.031250f, 0.500000f },
    float2{ 0.000000f, 0.593750f }, float2{ 0.031250f, 0.593750f }, float2{ 0.031250f, 0.562500f },
    float2{ 0.000000f, 0.656250f }, float2{ 0.031250f, 0.656250f }, float2{ 0.031250f, 0.625000f },
    float2{ 0.062500f, 0.656250f }, float2{ 0.093750f, 0.656250f }, float2{ 0.093750f, 0.625000f },
    float2{ 0.000000f, 0.718750f }, float2{ 0.031250f, 0.718750f }, float2{ 0.031250f, 0.687500f },
    float2{ 0.000000f, 0.781250f }, float2{ 0.031250f, 0.781250f }, float2{ 0.031250f, 0.750000f },
    float2{ 0.062500f, 0.781250f }, float2{ 0.093750f, 0.781250f }, float2{ 0.093750f, 0.750000f },
    float2{ 0.000000f, 0.843750f }, float2{ 0.031250f, 0.843750f }, float2{ 0.031250f, 0.812500f },
    float2{ 0.062500f, 0.843750f }, float2{ 0.093750f, 0.843750f }, float2{ 0.093750f, 0.812500f },
    float2{ 0.125000f, 0.781250f }, float2{ 0.156250f, 0.781250f }, float2{ 0.156250f, 0.750000f },
    float2{ 0.187500f, 0.781250f }, float2{ 0.218750f, 0.781250f }, float2{ 0.218750f, 0.750000f },
    float2{ 0.125000f, 0.843750f }, float2{ 0.156250f, 0.843750f }, float2{ 0.156250f, 0.812500f },
    float2{ 0.062500f, 0.906250f }, float2{ 0.093750f, 0.906250f }, float2{ 0.093750f, 0.875000f },
    float2{ 0.000000f, 0.906250f }, float2{ 0.031250f, 0.906250f }, float2{ 0.031250f, 0.875000f },
    float2{ 0.000000f, 0.968750f }, float2{ 0.031250f, 0.968750f }, float2{ 0.031250f, 0.937500f },
};

struct MicroVertexInfo {
    uint32_t adjA : 8; // to support the max vertex index 152
    uint32_t adjB : 8;
    uint32_t level : 3;
    uint32_t placeHolder : 13;
};

CUDA_CONSTANT_MEM constexpr MicroVertexInfo microVertexInfos[] = {
    { 255, 255, 0 }, { 255, 255, 0 }, { 255, 255, 0 },
    {   0,   2, 1 }, {   1,   2, 1 }, {   0,   1, 1 },
    {   0,   3, 2 }, {   5,   3, 2 }, {   0,   5, 2 },
    {   5,   4, 2 }, {   1,   4, 2 }, {   5,   1, 2 },
    {   3,   2, 2 }, {   4,   2, 2 }, {   4,   3, 2 },
    {   0,   6, 3 }, {   8,   6, 3 }, {   0,   8, 3 },
    {   8,   7, 3 }, {   5,   7, 3 }, {   8,   5, 3 },
    {   6,   3, 3 }, {   7,   3, 3 }, {   7,   6, 3 },
    {   7,  14, 3 }, {   9,  14, 3 }, {   7,   9, 3 },
    {   5,   9, 3 }, {  11,   9, 3 }, {   5,  11, 3 },
    {  11,  10, 3 }, {   1,  10, 3 }, {  11,   1, 3 },
    {   9,   4, 3 }, {  10,   4, 3 }, {  10,   9, 3 },
    {  14,  13, 3 }, {   4,  13, 3 }, {   4,  14, 3 },
    {   3,  12, 3 }, {  14,  12, 3 }, {  14,   3, 3 },
    {  12,   2, 3 }, {  13,   2, 3 }, {  12,  13, 3 },
    {   0,  15, 4 }, {  17,  15, 4 }, {   0,  17, 4 },
    {  17,  16, 4 }, {   8,  16, 4 }, {  17,   8, 4 },
    {  15,   6, 4 }, {  16,   6, 4 }, {  16,  15, 4 },
    {  16,  23, 4 }, {  18,  23, 4 }, {  16,  18, 4 },
    {   8,  18, 4 }, {  20,  18, 4 }, {   8,  20, 4 },
    {  20,  19, 4 }, {   5,  19, 4 }, {  20,   5, 4 },
    {  18,   7, 4 }, {  19,   7, 4 }, {  19,  18, 4 },
    {  23,  22, 4 }, {   7,  22, 4 }, {   7,  23, 4 },
    {   6,  21, 4 }, {  23,  21, 4 }, {  23,   6, 4 },
    {  21,   3, 4 }, {  22,   3, 4 }, {  21,  22, 4 },
    {  22,  41, 4 }, {  24,  41, 4 }, {  22,  24, 4 },
    {   7,  24, 4 }, {  26,  24, 4 }, {   7,  26, 4 },
    {  26,  25, 4 }, {   9,  25, 4 }, {  26,   9, 4 },
    {  24,  14, 4 }, {  25,  14, 4 }, {  25,  24, 4 },
    {  25,  38, 4 }, {  33,  38, 4 }, {  25,  33, 4 },
    {  19,  26, 4 }, {  27,  26, 4 }, {  27,  19, 4 },
    {   5,  27, 4 }, {  29,  27, 4 }, {   5,  29, 4 },
    {  29,  28, 4 }, {  11,  28, 4 }, {  29,  11, 4 },
    {  27,   9, 4 }, {  28,   9, 4 }, {  28,  27, 4 },
    {  28,  35, 4 }, {  30,  35, 4 }, {  28,  30, 4 },
    {  11,  30, 4 }, {  32,  30, 4 }, {  11,  32, 4 },
    {  32,  31, 4 }, {   1,  31, 4 }, {  32,   1, 4 },
    {  30,  10, 4 }, {  31,  10, 4 }, {  31,  30, 4 },
    {  35,  34, 4 }, {  10,  34, 4 }, {  10,  35, 4 },
    {   9,  33, 4 }, {  35,  33, 4 }, {  35,   9, 4 },
    {  33,   4, 4 }, {  34,   4, 4 }, {  33,  34, 4 },
    {  38,  37, 4 }, {   4,  37, 4 }, {   4,  38, 4 },
    {  14,  36, 4 }, {  38,  36, 4 }, {  38,  14, 4 },
    {  36,  13, 4 }, {  37,  13, 4 }, {  36,  37, 4 },
    {  40,  44, 4 }, {  36,  44, 4 }, {  36,  40, 4 },
    {  41,  40, 4 }, {  14,  40, 4 }, {  14,  41, 4 },
    {   3,  39, 4 }, {  41,  39, 4 }, {  41,   3, 4 },
    {  39,  12, 4 }, {  40,  12, 4 }, {  39,  40, 4 },
    {  12,  42, 4 }, {  44,  42, 4 }, {  12,  44, 4 },
    {  44,  43, 4 }, {  13,  43, 4 }, {  44,  13, 4 },
    {  42,   2, 4 }, {  43,   2, 4 }, {  43,  42, 4 },
    {   0,  45, 5 }, {  47,  45, 5 }, {   0,  47, 5 },
    {  47,  46, 5 }, {  17,  46, 5 }, {  47,  17, 5 },
    {  45,  15, 5 }, {  46,  15, 5 }, {  46,  45, 5 },
    {  46,  53, 5 }, {  48,  53, 5 }, {  46,  48, 5 },
    {  17,  48, 5 }, {  50,  48, 5 }, {  17,  50, 5 },
    {  50,  49, 5 }, {   8,  49, 5 }, {  50,   8, 5 },
    {  48,  16, 5 }, {  49,  16, 5 }, {  49,  48, 5 },
    {  53,  52, 5 }, {  16,  52, 5 }, {  16,  53, 5 },
    {  15,  51, 5 }, {  53,  51, 5 }, {  53,  15, 5 },
    {  51,   6, 5 }, {  52,   6, 5 }, {  51,  52, 5 },
    {  52,  71, 5 }, {  54,  71, 5 }, {  52,  54, 5 },
    {  16,  54, 5 }, {  56,  54, 5 }, {  16,  56, 5 },
    {  56,  55, 5 }, {  18,  55, 5 }, {  56,  18, 5 },
    {  54,  23, 5 }, {  55,  23, 5 }, {  55,  54, 5 },
    {  55,  68, 5 }, {  63,  68, 5 }, {  55,  63, 5 },
    {  49,  56, 5 }, {  57,  56, 5 }, {  57,  49, 5 },
    {   8,  57, 5 }, {  59,  57, 5 }, {   8,  59, 5 },
    {  59,  58, 5 }, {  20,  58, 5 }, {  59,  20, 5 },
    {  57,  18, 5 }, {  58,  18, 5 }, {  58,  57, 5 },
    {  58,  65, 5 }, {  60,  65, 5 }, {  58,  60, 5 },
    {  20,  60, 5 }, {  62,  60, 5 }, {  20,  62, 5 },
    {  62,  61, 5 }, {   5,  61, 5 }, {  62,   5, 5 },
    {  60,  19, 5 }, {  61,  19, 5 }, {  61,  60, 5 },
    {  65,  64, 5 }, {  19,  64, 5 }, {  19,  65, 5 },
    {  18,  63, 5 }, {  65,  63, 5 }, {  65,  18, 5 },
    {  63,   7, 5 }, {  64,   7, 5 }, {  63,  64, 5 },
    {  68,  67, 5 }, {   7,  67, 5 }, {   7,  68, 5 },
    {  23,  66, 5 }, {  68,  66, 5 }, {  68,  23, 5 },
    {  66,  22, 5 }, {  67,  22, 5 }, {  66,  67, 5 },
    {  70,  74, 5 }, {  66,  74, 5 }, {  66,  70, 5 },
    {  71,  70, 5 }, {  23,  70, 5 }, {  23,  71, 5 },
    {   6,  69, 5 }, {  71,  69, 5 }, {  71,   6, 5 },
    {  69,  21, 5 }, {  70,  21, 5 }, {  69,  70, 5 },
    {  21,  72, 5 }, {  74,  72, 5 }, {  21,  74, 5 },
    {  74,  73, 5 }, {  22,  73, 5 }, {  74,  22, 5 },
    {  72,   3, 5 }, {  73,   3, 5 }, {  73,  72, 5 },
    {  73, 140, 5 }, {  75, 140, 5 }, {  73,  75, 5 },
    {  22,  75, 5 }, {  77,  75, 5 }, {  22,  77, 5 },
    {  77,  76, 5 }, {  24,  76, 5 }, {  77,  24, 5 },
    {  75,  41, 5 }, {  76,  41, 5 }, {  76,  75, 5 },
    {  76, 137, 5 }, {  84, 137, 5 }, {  76,  84, 5 },
    {  67,  77, 5 }, {  78,  77, 5 }, {  78,  67, 5 },
    {   7,  78, 5 }, {  80,  78, 5 }, {   7,  80, 5 },
    {  80,  79, 5 }, {  26,  79, 5 }, {  80,  26, 5 },
    {  78,  24, 5 }, {  79,  24, 5 }, {  79,  78, 5 },
    {  79,  86, 5 }, {  81,  86, 5 }, {  79,  81, 5 },
    {  26,  81, 5 }, {  83,  81, 5 }, {  26,  83, 5 },
    {  83,  82, 5 }, {   9,  82, 5 }, {  83,   9, 5 },
    {  81,  25, 5 }, {  82,  25, 5 }, {  82,  81, 5 },
    {  86,  85, 5 }, {  25,  85, 5 }, {  25,  86, 5 },
    {  24,  84, 5 }, {  86,  84, 5 }, {  86,  24, 5 },
    {  84,  14, 5 }, {  85,  14, 5 }, {  84,  85, 5 },
    {  85, 128, 5 }, {  87, 128, 5 }, {  85,  87, 5 },
    {  25,  87, 5 }, {  89,  87, 5 }, {  25,  89, 5 },
    {  89,  88, 5 }, {  33,  88, 5 }, {  89,  33, 5 },
    {  87,  38, 5 }, {  88,  38, 5 }, {  88,  87, 5 },
    {  88, 125, 5 }, { 120, 125, 5 }, {  88, 120, 5 },
    {  82,  89, 5 }, { 117,  89, 5 }, { 117,  82, 5 },
    {  91,  83, 5 }, {  99,  83, 5 }, {  99,  91, 5 },
    {  92,  91, 5 }, {  27,  91, 5 }, {  27,  92, 5 },
    {  19,  90, 5 }, {  92,  90, 5 }, {  92,  19, 5 },
    {  90,  26, 5 }, {  91,  26, 5 }, {  90,  91, 5 },
    {  64,  80, 5 }, {  90,  80, 5 }, {  90,  64, 5 },
    {  61,  92, 5 }, {  93,  92, 5 }, {  61,  93, 5 },
    {   5,  93, 5 }, {  95,  93, 5 }, {   5,  95, 5 },
    {  95,  94, 5 }, {  29,  94, 5 }, {  95,  29, 5 },
    {  93,  27, 5 }, {  94,  27, 5 }, {  94,  93, 5 },
    {  94, 101, 5 }, {  96, 101, 5 }, {  94,  96, 5 },
    {  29,  96, 5 }, {  98,  96, 5 }, {  29,  98, 5 },
    {  98,  97, 5 }, {  11,  97, 5 }, {  98,  11, 5 },
    {  96,  28, 5 }, {  97,  28, 5 }, {  97,  96, 5 },
    { 101, 100, 5 }, {  28, 100, 5 }, {  28, 101, 5 },
    {  27,  99, 5 }, { 101,  99, 5 }, { 101,  27, 5 },
    {  99,   9, 5 }, { 100,   9, 5 }, {  99, 100, 5 },
    { 100, 119, 5 }, { 102, 119, 5 }, { 100, 102, 5 },
    {  28, 102, 5 }, { 104, 102, 5 }, {  28, 104, 5 },
    { 104, 103, 5 }, {  30, 103, 5 }, { 104,  30, 5 },
    { 102,  35, 5 }, { 103,  35, 5 }, { 103, 102, 5 },
    { 103, 116, 5 }, { 111, 116, 5 }, { 103, 111, 5 },
    {  97, 104, 5 }, { 105, 104, 5 }, { 105,  97, 5 },
    {  11, 105, 5 }, { 107, 105, 5 }, {  11, 107, 5 },
    { 107, 106, 5 }, {  32, 106, 5 }, { 107,  32, 5 },
    { 105,  30, 5 }, { 106,  30, 5 }, { 106, 105, 5 },
    { 106, 113, 5 }, { 108, 113, 5 }, { 106, 108, 5 },
    {  32, 108, 5 }, { 110, 108, 5 }, {  32, 110, 5 },
    { 110, 109, 5 }, {   1, 109, 5 }, { 110,   1, 5 },
    { 108,  31, 5 }, { 109,  31, 5 }, { 109, 108, 5 },
    { 113, 112, 5 }, {  31, 112, 5 }, {  31, 113, 5 },
    {  30, 111, 5 }, { 113, 111, 5 }, { 113,  30, 5 },
    { 111,  10, 5 }, { 112,  10, 5 }, { 111, 112, 5 },
    { 116, 115, 5 }, {  10, 115, 5 }, {  10, 116, 5 },
    {  35, 114, 5 }, { 116, 114, 5 }, { 116,  35, 5 },
    { 114,  34, 5 }, { 115,  34, 5 }, { 114, 115, 5 },
    { 118, 122, 5 }, { 114, 122, 5 }, { 114, 118, 5 },
    { 119, 118, 5 }, {  35, 118, 5 }, {  35, 119, 5 },
    {   9, 117, 5 }, { 119, 117, 5 }, { 119,   9, 5 },
    { 117,  33, 5 }, { 118,  33, 5 }, { 117, 118, 5 },
    {  33, 120, 5 }, { 122, 120, 5 }, {  33, 122, 5 },
    { 122, 121, 5 }, {  34, 121, 5 }, { 122,  34, 5 },
    { 120,   4, 5 }, { 121,   4, 5 }, { 121, 120, 5 },
    { 125, 124, 5 }, {   4, 124, 5 }, {   4, 125, 5 },
    {  38, 123, 5 }, { 125, 123, 5 }, { 125,  38, 5 },
    { 123,  37, 5 }, { 124,  37, 5 }, { 123, 124, 5 },
    { 127, 131, 5 }, { 123, 131, 5 }, { 123, 127, 5 },
    { 128, 127, 5 }, {  38, 127, 5 }, {  38, 128, 5 },
    {  14, 126, 5 }, { 128, 126, 5 }, { 128,  14, 5 },
    { 126,  36, 5 }, { 127,  36, 5 }, { 126, 127, 5 },
    {  36, 129, 5 }, { 131, 129, 5 }, {  36, 131, 5 },
    { 131, 130, 5 }, {  37, 130, 5 }, { 131,  37, 5 },
    { 129,  13, 5 }, { 130,  13, 5 }, { 130, 129, 5 },
    { 133, 149, 5 }, { 129, 149, 5 }, { 129, 133, 5 },
    { 134, 133, 5 }, {  36, 133, 5 }, {  36, 134, 5 },
    {  40, 132, 5 }, { 134, 132, 5 }, { 134,  40, 5 },
    { 132,  44, 5 }, { 133,  44, 5 }, { 132, 133, 5 },
    { 142, 146, 5 }, { 132, 146, 5 }, { 132, 142, 5 },
    { 136, 134, 5 }, { 126, 134, 5 }, { 136, 126, 5 },
    { 137, 136, 5 }, {  14, 136, 5 }, {  14, 137, 5 },
    {  41, 135, 5 }, { 137, 135, 5 }, { 137,  41, 5 },
    { 135,  40, 5 }, { 136,  40, 5 }, { 135, 136, 5 },
    { 139, 143, 5 }, { 135, 143, 5 }, { 135, 139, 5 },
    { 140, 139, 5 }, {  41, 139, 5 }, {  41, 140, 5 },
    {   3, 138, 5 }, { 140, 138, 5 }, { 140,   3, 5 },
    { 138,  39, 5 }, { 139,  39, 5 }, { 138, 139, 5 },
    {  39, 141, 5 }, { 143, 141, 5 }, {  39, 143, 5 },
    { 143, 142, 5 }, {  40, 142, 5 }, { 143,  40, 5 },
    { 141,  12, 5 }, { 142,  12, 5 }, { 142, 141, 5 },
    {  12, 144, 5 }, { 146, 144, 5 }, {  12, 146, 5 },
    { 146, 145, 5 }, {  44, 145, 5 }, { 146,  44, 5 },
    { 144,  42, 5 }, { 145,  42, 5 }, { 145, 144, 5 },
    { 145, 152, 5 }, { 147, 152, 5 }, { 145, 147, 5 },
    {  44, 147, 5 }, { 149, 147, 5 }, {  44, 149, 5 },
    { 149, 148, 5 }, {  13, 148, 5 }, { 149,  13, 5 },
    { 147,  43, 5 }, { 148,  43, 5 }, { 148, 147, 5 },
    { 152, 151, 5 }, {  43, 151, 5 }, {  43, 152, 5 },
    {  42, 150, 5 }, { 152, 150, 5 }, { 152,  42, 5 },
    { 150,   2, 5 }, { 151,   2, 5 }, { 150, 151, 5 },
};
#endif

namespace shared {
#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
    struct DisplacementBlockLayoutInfo {
        uint32_t correctionBitOffsets[6];
        uint32_t correctionBitWidths[6];
        uint32_t shiftBitOffsets[6];
        uint32_t shiftBitWidths[6];
        uint32_t maxShifts[6];
    };

    CUDA_CONSTANT_MEM DisplacementBlockLayoutInfo displacementBlock256MicroTrisLayoutInfo = {
        { 0, 33, 66, 165, 465, 0xFFFFFFFF },
        { 11, 11, 11, 10, 5, 0xFFFFFFFF },
        { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 1018, 1006, 0xFFFFFFFF },
        { 0, 0, 0, 1, 3, 0xFFFFFFFF },
        { 0, 0, 0, 1, 6, 0xFFFFFFFF },
    };
    CUDA_CONSTANT_MEM DisplacementBlockLayoutInfo displacementBlock1024MicroTrisLayoutInfo = {
        { 0, 33, 66, 138, 258, 474 },
        { 11, 11, 8, 4, 2, 1 },
        { 0xFFFFFFFF, 0xFFFFFFFF, 1014, 1002, 986, 970 },
        { 0, 0, 2, 3, 4, 4 },
        { 0, 0, 3, 7, 9, 10 },
    };

    template <OptixDisplacementMicromapFormat encoding>
    CUDA_DEVICE_FUNCTION const DisplacementBlockLayoutInfo &getDisplacementBlockLayoutInfo() {
        if constexpr (encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES)
            return displacementBlock256MicroTrisLayoutInfo;
        else /*if constexpr (OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES)*/
            return displacementBlock1024MicroTrisLayoutInfo;
    }
#endif



    template <OptixDisplacementMicromapFormat encoding>
    struct DisplacementBlock;

    template <>
    struct DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES> {
        static constexpr uint32_t maxSubdivLevel = 3;
        static constexpr uint32_t maxNumMicroTris = 1 << (2 * maxSubdivLevel);
        static constexpr uint32_t maxNumEdgeVertices = (1 << maxSubdivLevel) + 1;
        static constexpr uint32_t maxNumMicroVertices = (1 + maxNumEdgeVertices) * maxNumEdgeVertices / 2;
        static constexpr uint32_t numBytes = 64;
        static constexpr uint32_t numDwords = numBytes / sizeof(uint32_t);

        static constexpr uint32_t numBitsPerValue = 11;
        static constexpr uint32_t maxValue = (1 << numBitsPerValue) - 1;

        uint32_t data[numDwords];

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION void setValue(uint32_t microVtxIdx, float value) {
            Assert(value <= 1.0f, "Height value must be normalized: %g", value);
            constexpr uint32_t _numBitsPerValue = numBitsPerValue;
            constexpr uint32_t _maxValue = maxValue; // workaround for NVCC bug? (CUDA 11.7)
            const uint32_t uiValue = min(static_cast<uint32_t>(maxValue * value), _maxValue);
            const uint32_t bitOffset = numBitsPerValue * microVtxIdx;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, _numBitsPerValue);
            atomicOr(&data[binIdx], (uiValue & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < numBitsPerValue)
                atomicOr(&data[binIdx + 1], uiValue >> numLowerBits);
        }
#endif

        float getValue(uint32_t microVtxIdx) const {
            const uint32_t bitOffset = numBitsPerValue * microVtxIdx;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, numBitsPerValue);
            uint32_t uiValue = 0;
            uiValue |= ((data[binIdx] >> bitOffsetInBin) & ((1 << numLowerBits) - 1));
            if (numLowerBits < numBitsPerValue)
                uiValue |= (data[binIdx + 1] & ((1 << (numBitsPerValue - numLowerBits)) - 1)) << numLowerBits;
            return static_cast<float>(uiValue) / maxValue;
        }
    };

    template <>
    struct DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES> {
        static constexpr uint32_t maxSubdivLevel = 4;
        static constexpr uint32_t maxNumMicroTris = 1 << (2 * maxSubdivLevel);
        static constexpr uint32_t numBytes = 128;
        static constexpr uint32_t numDwords = numBytes / sizeof(uint32_t);

        static constexpr uint32_t anchorBitOffset = 0;
        static constexpr uint32_t anchorBitWidth = 11;
        static constexpr uint32_t maxAnchorValue = (1 << anchorBitWidth) - 1;

        static constexpr uint32_t correctionBitOffsets[] = {
            0, 33, 66, 165, 465, 0xFFFFFFFF
        };
        static constexpr uint32_t correctionBitWidths[] = {
            11, 11, 11, 10, 5, 0xFFFFFFFF
        };
        static constexpr uint32_t shiftBitOffsets[] = {
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 1018, 1006, 0xFFFFFFFF
        };
        static constexpr uint32_t shiftBitWidths[] = {
            0, 0, 0, 1, 3, 0xFFFFFFFF
        };
        static constexpr uint32_t maxShifts[] = {
            0, 0, 0, 1, 6, 0xFFFFFFFF
        };

        uint32_t data[numDwords];

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t quantize(float value) {
            Assert(value <= 1.0f, "Height value must be normalized: %g", value);
            constexpr uint32_t _maxAnchorValue = maxAnchorValue; // workaround for NVCC bug? (CUDA 11.7)
            const uint32_t uiValue = min(static_cast<uint32_t>(maxAnchorValue * value), _maxAnchorValue);
            return uiValue;
        }
        CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t dequantize(uint32_t value) {
            Assert(value <= maxAnchorValue, "OOB quantized value: %u", value);
            const float fValue = static_cast<float>(value) / maxAnchorValue;
            return fValue;
        }

        CUDA_DEVICE_FUNCTION void setValue(uint32_t level, uint32_t microVtxIdxInLevel, uint32_t value) {
            const DisplacementBlockLayoutInfo &layoutInfo = displacementBlock256MicroTrisLayoutInfo;
            const uint32_t bitWidth = layoutInfo.correctionBitWidths[level];
            const uint32_t mask = (1 << bitWidth) - 1;
            value &= mask;
            const uint32_t bitOffset = layoutInfo.correctionBitOffsets[level] + bitWidth * microVtxIdxInLevel;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            atomicOr(&data[binIdx], (value & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < bitWidth)
                atomicOr(&data[binIdx + 1], value >> numLowerBits);
        }

        CUDA_DEVICE_FUNCTION void setShift(uint32_t level, uint32_t vtxType, uint32_t value) {
            const DisplacementBlockLayoutInfo &layoutInfo = displacementBlock256MicroTrisLayoutInfo;
            const uint32_t bitWidth = layoutInfo.shiftBitWidths[level];
            const uint32_t mask = (1 << bitWidth) - 1;
            value &= mask;
            const uint32_t bitOffset = layoutInfo.shiftBitOffsets[level] + bitWidth * vtxType;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            atomicOr(&data[binIdx], (value & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < bitWidth)
                atomicOr(&data[binIdx + 1], value >> numLowerBits);
        }
#endif
    };

    template <>
    struct DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES> {
        static constexpr uint32_t maxSubdivLevel = 5;
        static constexpr uint32_t maxNumMicroTris = 1 << (2 * maxSubdivLevel);
        static constexpr uint32_t numBytes = 128;
        static constexpr uint32_t numDwords = numBytes / sizeof(uint32_t);

        static constexpr uint32_t anchorBitOffset = 0;
        static constexpr uint32_t anchorBitWidth = 11;
        static constexpr uint32_t maxAnchorValue = (1 << anchorBitWidth) - 1;

        static constexpr uint32_t correctionBitOffsets[] = {
            0, 33, 66, 138, 258, 474
        };
        static constexpr uint32_t correctionBitWidths[] = {
            11, 11, 8, 4, 2, 1
        };
        static constexpr uint32_t shiftBitOffsets[] = {
            0xFFFFFFFF, 0xFFFFFFFF, 1014, 1002, 986, 970
        };
        static constexpr uint32_t shiftBitWidths[] = {
            0, 0, 2, 3, 4, 4
        };
        static constexpr uint32_t maxShifts[] = {
            0, 0, 3, 7, 9, 10
        };

        uint32_t data[numDwords];

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t quantize(float value) {
            Assert(value <= 1.0f, "Height value must be normalized: %g", value);
            constexpr uint32_t _maxAnchorValue = maxAnchorValue; // workaround for NVCC bug? (CUDA 11.7)
            const uint32_t uiValue = min(static_cast<uint32_t>(maxAnchorValue * value), _maxAnchorValue);
            return uiValue;
        }
        CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t dequantize(uint32_t value) {
            Assert(value <= maxAnchorValue, "OOB quantized value: %u", value);
            const float fValue = static_cast<float>(value) / maxAnchorValue;
            return fValue;
        }

        CUDA_DEVICE_FUNCTION void setValue(uint32_t level, uint32_t microVtxIdxInLevel, uint32_t value) {
            const DisplacementBlockLayoutInfo &layoutInfo = displacementBlock1024MicroTrisLayoutInfo;
            const uint32_t bitWidth = layoutInfo.correctionBitWidths[level];
            const uint32_t mask = (1 << bitWidth) - 1;
            value &= mask;
            const uint32_t bitOffset = layoutInfo.correctionBitOffsets[level] + bitWidth * microVtxIdxInLevel;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            atomicOr(&data[binIdx], (value & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < bitWidth)
                atomicOr(&data[binIdx + 1], value >> numLowerBits);
        }

        CUDA_DEVICE_FUNCTION void setShift(uint32_t level, uint32_t vtxType, uint32_t value) {
            const DisplacementBlockLayoutInfo &layoutInfo = displacementBlock1024MicroTrisLayoutInfo;
            const uint32_t bitWidth = layoutInfo.shiftBitWidths[level];
            const uint32_t mask = (1 << bitWidth) - 1;
            value &= mask;
            const uint32_t bitOffset = layoutInfo.shiftBitOffsets[level] + bitWidth * vtxType;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            atomicOr(&data[binIdx], (value & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < bitWidth)
                atomicOr(&data[binIdx + 1], value >> numLowerBits);
        }
#endif

        CUDA_COMMON_FUNCTION void setBits(uint32_t value, uint32_t bitOffset, uint32_t bitWidth) {
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            const uint32_t lowerMask = (1 << numLowerBits) - 1;
            data[binIdx] &= ~(lowerMask << bitOffsetInBin);
            data[binIdx] |= (value & lowerMask) << bitOffsetInBin;
            if (numLowerBits < bitWidth) {
                const uint32_t higherMask = (1 << (bitWidth - numLowerBits)) - 1;
                data[binIdx + 1] &= ~higherMask;
                data[binIdx + 1] |= value >> numLowerBits;
            }
        }

        CUDA_COMMON_FUNCTION uint32_t getBits(uint32_t bitOffset, uint32_t bitWidth) const {
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            uint32_t value = 0;
            value |= ((data[binIdx] >> bitOffsetInBin) & ((1 << numLowerBits) - 1));
            if (numLowerBits < bitWidth)
                value |= (data[binIdx + 1] & ((1 << (bitWidth - numLowerBits)) - 1)) << numLowerBits;
            return value;
        }
    };
}

#if !defined(__CUDA_ARCH__)

struct Context {
    shared::StridedBuffer<float3> positions;
    shared::StridedBuffer<float2> texCoords;
    shared::StridedBuffer<shared::Triangle> triangles;
    CUtexObject texture;
    uint2 texSize;
    uint32_t numChannels;
    uint32_t alphaChannelIndex;
    shared::DMMEncoding maxCompressedFormat;
    shared::DMMSubdivLevel minSubdivLevel;
    shared::DMMSubdivLevel maxSubdivLevel;
    uint32_t subdivLevelBias;
    bool useIndexBuffer;
    uint32_t indexSize;
    CUdeviceptr scratchMem;
    size_t scratchMemSize;

    shared::DirectedEdge* directedEdges;
    uint32_t* halfEdgeIndices;
    shared::HalfEdge* halfEdges;
    CUdeviceptr memForSortDirectedEdges;
    size_t memSizeForSortDirectedEdges;
    shared::TriNeighborList* triNeighborLists;

    AABBAsOrderedInt* meshAabbAsOrderedInt;
    AABB* meshAabb;
    float* meshAabbArea;
    shared::MicroMapKey* microMapKeys;
    shared::MicroMapFormat* microMapFormats;
    uint32_t* triIndices;
    CUdeviceptr memForSortMicroMapKeys;
    size_t memSizeForSortMicroMapKeys;
    uint32_t* refKeyIndices;
    CUdeviceptr memForScanRefKeyIndices;
    size_t memSizeForScanRefKeyIndices;

    uint64_t* dmmSizes;
    uint32_t* hasDmmFlags;
    uint32_t* histInDmmArray;
    uint32_t* histInMesh;
    CUdeviceptr memForScanDmmSizes;
    size_t memSizeForScanDmmSizes;
    CUdeviceptr memForScanHasDmmFlags;
    size_t memSizeForScanHasDmmFlags;
    uint32_t* counter;
};

#endif
