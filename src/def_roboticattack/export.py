"""Export pipeline: pth → safetensors → ONNX → TRT FP16 + TRT FP32.

Usage:
    python -m def_roboticattack.export --checkpoint /path/to/best.pth --output-dir /path/to/exports
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from def_roboticattack.models.patch_detector import PatchDetectorNet


def export_safetensors(model: PatchDetectorNet, output_dir: Path) -> Path:
    from safetensors.torch import save_file

    state = model.state_dict()
    path = output_dir / "patch_detector.safetensors"
    save_file(state, str(path))
    print(f"[EXPORT] safetensors → {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def export_onnx(model: PatchDetectorNet, output_dir: Path, image_size: int = 224) -> Path:
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    path = output_dir / "patch_detector.onnx"

    # Use legacy exporter to get single-file ONNX (no external data)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=18,
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        dynamo=False,
    )

    # Remove any external data file if it was created
    data_path = Path(str(path) + ".data")
    if data_path.exists():
        # Re-save as single file
        import onnx

        onnx_model = onnx.load(str(path), load_external_data=True)
        onnx.save(onnx_model, str(path), save_as_external_data=False)
        data_path.unlink(missing_ok=True)

    print(f"[EXPORT] ONNX → {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def export_trt(onnx_path: Path, output_dir: Path, fp16: bool = True) -> Path | None:
    """Export ONNX → TensorRT using shared toolkit, trtexec, or TRT Python API."""
    suffix = "fp16" if fp16 else "fp32"
    stem = onnx_path.stem
    trt_path = output_dir / f"{stem}_{suffix}.trt"

    # Try shared TRT toolkit first (produces {stem}_fp16.trt and {stem}_fp32.trt)
    shared_trt = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if shared_trt.exists():
        cmd = [
            "python", str(shared_trt),
            "--onnx", str(onnx_path),
            "--output-dir", str(output_dir),
        ]
        # The toolkit exports both fp16 and fp32 by default; use --no-fp16/--no-fp32 to skip
        if not fp16:
            cmd.append("--no-fp16")
        if fp16:
            cmd.append("--no-fp32")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and trt_path.exists():
                print(f"[EXPORT] TRT {suffix} → {trt_path} ({trt_path.stat().st_size / 1024:.1f} KB)")
                return trt_path
            print(f"[WARN] Shared TRT toolkit: {result.stderr[:300]}")
        except Exception as e:
            print(f"[WARN] Shared TRT toolkit error: {e}")

    # Fallback: try trtexec directly
    trtexec = shutil.which("trtexec")
    if trtexec:
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={trt_path}",
            "--minShapes=image:1x3x224x224",
            "--optShapes=image:8x3x224x224",
            "--maxShapes=image:32x3x224x224",
        ]
        if fp16:
            cmd.append("--fp16")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and trt_path.exists():
                print(f"[EXPORT] TRT {suffix} → {trt_path} ({trt_path.stat().st_size / 1024:.1f} KB)")
                return trt_path
            print(f"[WARN] trtexec failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"[WARN] trtexec error: {e}")

    # Fallback: use TensorRT Python API
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT PARSE ERROR] {parser.get_error(i)}")
                return None

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        profile.set_shape("image", (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes:
            with open(trt_path, "wb") as f:
                f.write(engine_bytes)
            print(f"[EXPORT] TRT {suffix} → {trt_path} ({trt_path.stat().st_size / 1024:.1f} KB)")
            return trt_path
    except ImportError:
        print(f"[SKIP] TensorRT not available for {suffix} export")
    except Exception as e:
        print(f"[WARN] TRT Python API failed: {e}")

    return None


def run_export(checkpoint_path: str, output_dir: str, image_size: int = 224):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchDetectorNet(in_channels=3).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # 1. Save raw pth
    pth_path = output_dir / "patch_detector.pth"
    torch.save({"model": model.state_dict()}, pth_path)
    print(f"[EXPORT] pth → {pth_path} ({pth_path.stat().st_size / 1024:.1f} KB)")

    # 2. safetensors
    export_safetensors(model, output_dir)

    # 3. ONNX
    onnx_path = export_onnx(model, output_dir, image_size)

    # 4. TRT FP16
    export_trt(onnx_path, output_dir, fp16=True)

    # 5. TRT FP32
    export_trt(onnx_path, output_dir, fp16=False)

    print(f"\n[DONE] All exports in {output_dir}")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="/mnt/artifacts-datai/exports/DEF-roboticattack")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    run_export(args.checkpoint, args.output_dir, args.image_size)


if __name__ == "__main__":
    main()
