/**
 * DEF-roboticattack: VLA Robotic Attack CUDA Kernels
 * 1. fused_patch_apply — Apply adversarial patch to image at position with affine
 * 2. fused_action_perturb — Perturb robot action space with projected gradient
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void patch_apply_kernel(
    const float* __restrict__ image,
    const float* __restrict__ patch,
    float* __restrict__ output,
    int C, int H, int W,
    int pH, int pW,
    int pos_y, int pos_x
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * H * W) return;
    int c = idx / (H * W);
    int hw = idx % (H * W);
    int h = hw / W, w = hw % W;
    
    float val = image[idx];
    if (h >= pos_y && h < pos_y + pH && w >= pos_x && w < pos_x + pW) {
        int ph = h - pos_y, pw = w - pos_x;
        val = patch[c * pH * pW + ph * pW + pw];
    }
    output[idx] = val;
}

__global__ void action_perturb_kernel(
    const float* __restrict__ actions,
    const float* __restrict__ grads,
    float* __restrict__ output,
    float step_size, float eps,
    int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    float a = actions[idx];
    float g = grads[idx];
    float pert = a + step_size * ((g > 0) ? 1.0f : -1.0f);
    // Project to eps-ball
    float delta = pert - a;
    delta = fminf(fmaxf(delta, -eps), eps);
    output[idx] = a + delta;
}

torch::Tensor fused_patch_apply(
    torch::Tensor image, torch::Tensor patch, int pos_y, int pos_x
) {
    TORCH_CHECK(image.is_cuda(), "must be CUDA");
    auto output = torch::empty_like(image);
    int C=image.size(0), H=image.size(1), W=image.size(2);
    int pH=patch.size(1), pW=patch.size(2);
    int N = C * H * W;
    patch_apply_kernel<<<(N+255)/256, 256>>>(
        image.data_ptr<float>(), patch.data_ptr<float>(),
        output.data_ptr<float>(), C, H, W, pH, pW, pos_y, pos_x);
    return output;
}

torch::Tensor fused_action_perturb(
    torch::Tensor actions, torch::Tensor grads, float step_size, float eps
) {
    TORCH_CHECK(actions.is_cuda(), "must be CUDA");
    auto output = torch::empty_like(actions);
    int N = actions.numel();
    action_perturb_kernel<<<(N+255)/256, 256>>>(
        actions.data_ptr<float>(), grads.data_ptr<float>(),
        output.data_ptr<float>(), step_size, eps, actions.size(0), actions.size(1));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_patch_apply", &fused_patch_apply, "Apply adversarial patch to image (CUDA)");
    m.def("fused_action_perturb", &fused_action_perturb, "PGD action perturbation (CUDA)");
}
