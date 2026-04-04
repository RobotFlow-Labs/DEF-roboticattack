#include <metal_stdlib>
using namespace metal;

// ANIMA IP Scaffold: MLX/Metal fused patch blend kernel.
kernel void fused_patch_blend(
    device const float* image [[buffer(0)]],
    device const float* patch [[buffer(1)]],
    device const float* mask [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float m = mask[gid];
    output[gid] = (1.0f - m) * image[gid] + m * patch[gid];
}
