#include "micro_map_generator_private.h"

static CUmodule s_mmModule;
cudau::Kernel g_computeMeshAABB;
cudau::Kernel g_finalizeMeshAABB;
cudau::Kernel g_initializeHalfEdges;
cudau::Kernel g_findTwinHalfEdges;
cudau::Kernel g_findTriangleNeighbors;
cudau::Kernel g_extractTexCoords;
cudau::Kernel g_testIfTCTupleIsUnique;
cudau::Kernel g_testIfMicroMapKeyIsUnique;

void initializeMicroMapGeneratorKernels(const std::filesystem::path &ptxDirPath) {
    static bool isInitialized = false;
    if (!isInitialized) {
        CUDADRV_CHECK(cuModuleLoad(
            &s_mmModule,
            (ptxDirPath / "micro_map_kernels.ptx").string().c_str()));
        g_computeMeshAABB.set(
            s_mmModule, "computeMeshAABB", cudau::dim3(32), 0);
        g_finalizeMeshAABB.set(
            s_mmModule, "finalizeMeshAABB", cudau::dim3(32), 0);
        g_initializeHalfEdges.set(
            s_mmModule, "initializeHalfEdges", cudau::dim3(32), 0);
        g_findTwinHalfEdges.set(
            s_mmModule, "findTwinHalfEdges", cudau::dim3(32), 0);
        g_findTriangleNeighbors.set(
            s_mmModule, "findTriangleNeighbors", cudau::dim3(32), 0);
        g_extractTexCoords.set(
            s_mmModule, "extractTexCoords", cudau::dim3(32), 0);
        g_testIfTCTupleIsUnique.set(
            s_mmModule, "testIfTCTupleIsUnique", cudau::dim3(32), 0);
        g_testIfMicroMapKeyIsUnique.set(
            s_mmModule, "testIfMicroMapKeyIsUnique", cudau::dim3(32), 0);

        isInitialized = true;
    }
}
