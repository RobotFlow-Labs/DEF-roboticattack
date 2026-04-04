from __future__ import annotations

import numpy as np
import typer

from def_roboticattack.pipeline.runtime import DefenseRuntime

app = typer.Typer(help="DEF-roboticattack CLI")


@app.command("backend-info")
def backend_info(backend: str = typer.Option("auto", help="auto|cuda|mlx|cpu")):
    runtime = DefenseRuntime(backend=backend)
    info = runtime.backend_info
    typer.echo(f"backend={info.name} torch_device={info.torch_device} supports_half={info.supports_half}")
    typer.echo(f"reason={info.reason}")


@app.command("dry-run")
def dry_run(
    backend: str = typer.Option("auto", help="auto|cuda|mlx|cpu"),
    batch_size: int = typer.Option(4, min=1),
    height: int = typer.Option(224, min=16),
    width: int = typer.Option(224, min=16),
):
    runtime = DefenseRuntime(backend=backend)

    if runtime.backend_info.name == "cuda":
        try:
            import torch

            device = runtime.backend_info.torch_device
            images = torch.rand(batch_size, 3, height, width, device=device)
        except Exception:
            images = np.random.rand(batch_size, 3, height, width).astype(np.float32)
    else:
        images = np.random.rand(batch_size, 3, height, width).astype(np.float32)

    _, detection = runtime.sanitize_and_score(images)
    risk = runtime.aggregate_risk(detection)

    typer.echo(f"samples={len(detection.score)} risk={risk:.4f}")
    typer.echo(f"flagged={sum(detection.flagged)}/{len(detection.flagged)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
