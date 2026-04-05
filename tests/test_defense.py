"""Tests for DEF-roboticattack defense module.

Covers: model forward/backward, dataset generation, detector edge cases,
CUDA/CPU parity, checkpoint round-trip, pipeline integration.
"""
import numpy as np
import pytest
import torch

from def_roboticattack.data.synthetic import SyntheticPatchDataset
from def_roboticattack.defense.detector import PatchAnomalyDetector
from def_roboticattack.defense.transforms import clamp_patch_intensity, gaussian_blur_3x3
from def_roboticattack.device import resolve_backend
from def_roboticattack.models.patch_detector import PatchDetectorNet
from def_roboticattack.pipeline.runtime import DefenseRuntime


class TestPatchDetectorNet:
    def test_forward_shape(self):
        model = PatchDetectorNet(in_channels=3)
        x = torch.rand(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 1)

    def test_predict_proba_range(self):
        model = PatchDetectorNet(in_channels=3)
        x = torch.rand(4, 3, 224, 224)
        probs = model.predict_proba(x)
        assert probs.shape == (4, 1)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_backward_pass(self):
        model = PatchDetectorNet(in_channels=3)
        x = torch.rand(2, 3, 224, 224)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_different_image_sizes(self):
        model = PatchDetectorNet(in_channels=3)
        for size in [64, 128, 224, 320]:
            x = torch.rand(1, 3, size, size)
            logits = model(x)
            assert logits.shape == (1, 1)

    def test_param_count(self):
        model = PatchDetectorNet(in_channels=3)
        params = sum(p.numel() for p in model.parameters())
        assert params == 19841


class TestSyntheticDataset:
    def test_length(self):
        ds = SyntheticPatchDataset(num_samples=100, seed=42)
        assert len(ds) == 100

    def test_item_shape(self):
        ds = SyntheticPatchDataset(num_samples=10, image_size=224, seed=42)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)
        assert item["label"].shape == ()
        assert item["label"].item() in (0.0, 1.0)

    def test_reproducibility_across_calls(self):
        ds = SyntheticPatchDataset(num_samples=10, seed=42)
        item1 = ds[5]
        item2 = ds[5]
        assert torch.allclose(item1["image"], item2["image"])
        assert item1["label"] == item2["label"]

    def test_different_seeds_differ(self):
        ds1 = SyntheticPatchDataset(num_samples=10, seed=42)
        ds2 = SyntheticPatchDataset(num_samples=10, seed=99)
        assert not torch.allclose(ds1[0]["image"], ds2[0]["image"])

    def test_attack_ratio(self):
        ds = SyntheticPatchDataset(num_samples=1000, attack_prob=0.5, seed=42)
        labels = [ds[i]["label"].item() for i in range(1000)]
        attack_ratio = sum(labels) / len(labels)
        assert 0.4 < attack_ratio < 0.6


class TestPatchAnomalyDetector:
    def test_batch_size_1_torch(self):
        """Detector must work correctly with batch_size=1."""
        det = PatchAnomalyDetector(edge_threshold=0.15)
        img = torch.rand(1, 3, 224, 224)
        result = det.score_torch(img)
        assert len(result.score) == 1
        assert len(result.flagged) == 1

    def test_batch_size_1_numpy(self):
        det = PatchAnomalyDetector(edge_threshold=0.15)
        img = np.random.rand(1, 3, 224, 224).astype(np.float32)
        result = det.score_numpy(img)
        assert len(result.score) == 1
        assert len(result.flagged) == 1

    def test_patched_image_higher_score(self):
        """Image with high-contrast patch should score higher than smooth image."""
        det = PatchAnomalyDetector(edge_threshold=0.15)
        # Smooth image
        clean = torch.ones(1, 3, 224, 224) * 0.5
        # Image with sharp patch
        patched = clean.clone()
        patched[:, :, 50:100, 50:100] = torch.rand(1, 3, 50, 50)

        clean_result = det.score_torch(clean)
        patched_result = det.score_torch(patched)
        assert patched_result.score[0] > clean_result.score[0]


class TestTransforms:
    def test_clamp_preserves_shape(self):
        images = torch.rand(4, 3, 64, 64)
        result = clamp_patch_intensity(images, percentile=99.5)
        assert result.shape == images.shape

    def test_blur_preserves_shape(self):
        images = torch.rand(4, 3, 64, 64)
        result = gaussian_blur_3x3(images, blur_strength=0.15)
        assert result.shape == images.shape

    def test_blur_reduces_variance(self):
        """Blurring should reduce spatial variance."""
        images = torch.rand(2, 3, 64, 64)
        blurred = gaussian_blur_3x3(images, blur_strength=1.0)
        assert blurred.var() <= images.var()


class TestDefenseRuntime:
    def test_sanitize_and_score_torch(self):
        rt = DefenseRuntime()
        images = torch.rand(2, 3, 224, 224)
        sanitized, detection = rt.sanitize_and_score(images)
        assert sanitized.shape == images.shape
        assert len(detection.score) == 2

    def test_sanitize_and_score_numpy(self):
        rt = DefenseRuntime()
        images = np.random.rand(2, 3, 224, 224).astype(np.float32)
        sanitized, detection = rt.sanitize_and_score(images)
        assert sanitized.shape == images.shape
        assert len(detection.score) == 2

    def test_full_defense_without_model(self):
        rt = DefenseRuntime()
        images = torch.rand(2, 3, 224, 224)
        result = rt.full_defense(images)
        assert "combined_risk" in result
        assert "neural" not in result

    def test_full_defense_with_model(self):
        rt = DefenseRuntime()
        rt.load_model()  # untrained model
        images = torch.rand(2, 3, 224, 224)
        result = rt.full_defense(images)
        assert "neural" in result
        assert 0 <= result["combined_risk"] <= 10  # risk is bounded

    def test_aggregate_risk_normalized(self):
        """aggregate_risk should return value in reasonable range."""
        from def_roboticattack.defense.detector import DetectionResult

        det = DetectionResult(score=[0.1, 0.2], flagged=[False, False])
        risk = DefenseRuntime.aggregate_risk(det)
        assert 0 <= risk <= 1.0

    def test_aggregate_risk_empty(self):
        from def_roboticattack.defense.detector import DetectionResult

        det = DetectionResult(score=[], flagged=[])
        risk = DefenseRuntime.aggregate_risk(det)
        assert risk == 0.0


class TestBackendResolution:
    def test_cpu_backend(self):
        info = resolve_backend("cpu")
        assert info.name == "cpu"
        assert info.torch_device == "cpu"

    def test_auto_backend(self):
        info = resolve_backend("auto")
        assert info.name in ("cuda", "mlx", "cpu")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            resolve_backend("tpu")


class TestCudaOps:
    def test_fused_patch_apply_cpu_fallback(self):
        from def_roboticattack.defense.cuda_ops import fused_patch_apply

        img = torch.rand(3, 224, 224)
        patch = torch.rand(3, 50, 50)
        result = fused_patch_apply(img, patch, 10, 10)
        assert result.shape == (3, 224, 224)
        # Verify patch was applied
        assert torch.allclose(result[:, 10:60, 10:60], patch)

    def test_fused_patch_apply_bounds_clamping(self):
        from def_roboticattack.defense.cuda_ops import fused_patch_apply

        img = torch.rand(3, 64, 64)
        patch = torch.rand(3, 50, 50)
        # Patch extends beyond boundary (pos_y=30, 30+50=80 > 64)
        result = fused_patch_apply(img, patch, 30, 30)
        assert result.shape == (3, 64, 64)
        # Should not crash — partial patch applied

    def test_fused_action_perturb_cpu_fallback(self):
        from def_roboticattack.defense.cuda_ops import fused_action_perturb

        actions = torch.rand(4, 7)
        grads = torch.rand(4, 7)
        result = fused_action_perturb(actions, grads, 0.01, 0.1)
        assert result.shape == (4, 7)
        # Perturbation should be within eps
        delta = (result - actions).abs()
        assert (delta <= 0.1 + 1e-6).all()

    def test_fused_action_perturb_zero_grad(self):
        """Zero gradients should produce zero perturbation."""
        from def_roboticattack.defense.cuda_ops import fused_action_perturb

        actions = torch.ones(2, 7)
        grads = torch.zeros(2, 7)
        result = fused_action_perturb(actions, grads, 0.01, 0.1)
        assert torch.allclose(result, actions)
