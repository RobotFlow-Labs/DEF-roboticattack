// ANIMA IP Scaffold: CUDA fused patch blend + normalize kernel.
// This file is a scaffold entrypoint for future optimized integration.

#include <cuda_runtime.h>

extern "C" __global__ void fused_patch_blend_kernel(
    const float* image,
    const float* patch,
    const float* mask,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m = mask[idx];
        output[idx] = (1.0f - m) * image[idx] + m * patch[idx];
    }
}

extern "C" void launch_fused_patch_blend(
    const float* image,
    const float* patch,
    const float* mask,
    float* output,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fused_patch_blend_kernel<<<blocks, threads, 0, stream>>>(image, patch, mask, output, n);
}
