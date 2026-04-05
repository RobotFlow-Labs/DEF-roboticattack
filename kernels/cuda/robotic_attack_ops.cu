/**
 * DEF-roboticattack: VLA Robotic Attack CUDA Kernels
 * 1. fused_patch_apply — Apply adversarial patch to image at position with bounds checking
 * 2. fused_action_perturb — Perturb robot action space with projected gradient (sign-correct)
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
    int pos_y, int pos_x,
    int clamp_pH, int clamp_pW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * H * W) return;
    int c = idx / (H * W);
    int hw = idx % (H * W);
    int h = hw / W, w = hw % W;

    float val = image[idx];
    if (h >= pos_y && h < pos_y + clamp_pH && w >= pos_x && w < pos_x + clamp_pW) {
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
    // Use sign function: 0 gradient -> 0 perturbation (matches torch.sign behavior)
    float sign_g = (g > 0.0f) ? 1.0f : ((g < 0.0f) ? -1.0f : 0.0f);
    float pert = a + step_size * sign_g;
    // Project to eps-ball
    float delta = pert - a;
    delta = fminf(fmaxf(delta, -eps), eps);
    output[idx] = a + delta;
}

torch::Tensor fused_patch_apply(
    torch::Tensor image, torch::Tensor patch, int pos_y, int pos_x
) {
    TORCH_CHECK(image.is_cuda(), "image must be CUDA tensor");
    TORCH_CHECK(patch.is_cuda(), "patch must be CUDA tensor");
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "image must be float32 (not fp16)");
    TORCH_CHECK(patch.scalar_type() == torch::kFloat32, "patch must be float32 (not fp16)");

    auto output = torch::empty_like(image);
    int C=image.size(0), H=image.size(1), W=image.size(2);
    int pH=patch.size(1), pW=patch.size(2);

    // Clamp patch region to image boundaries
    int clamp_pH = min(pH, H - pos_y);
    int clamp_pW = min(pW, W - pos_x);
    if (clamp_pH <= 0 || clamp_pW <= 0) {
        // Patch entirely outside image — return copy
        output.copy_(image);
        return output;
    }

    int N = C * H * W;
    patch_apply_kernel<<<(N+255)/256, 256>>>(
        image.data_ptr<float>(), patch.data_ptr<float>(),
        output.data_ptr<float>(), C, H, W, pH, pW, pos_y, pos_x, clamp_pH, clamp_pW);
    return output;
}

torch::Tensor fused_action_perturb(
    torch::Tensor actions, torch::Tensor grads, float step_size, float eps
) {
    TORCH_CHECK(actions.is_cuda(), "actions must be CUDA tensor");
    TORCH_CHECK(grads.is_cuda(), "grads must be CUDA tensor");
    TORCH_CHECK(actions.scalar_type() == torch::kFloat32, "actions must be float32");
    TORCH_CHECK(grads.scalar_type() == torch::kFloat32, "grads must be float32");

    auto output = torch::empty_like(actions);
    int N = actions.numel();
    action_perturb_kernel<<<(N+255)/256, 256>>>(
        actions.data_ptr<float>(), grads.data_ptr<float>(),
        output.data_ptr<float>(), step_size, eps, actions.size(0), actions.size(1));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_patch_apply", &fused_patch_apply, "Apply adversarial patch to image (CUDA, float32 only)");
    m.def("fused_action_perturb", &fused_action_perturb, "PGD action perturbation (CUDA, float32 only)");
}
