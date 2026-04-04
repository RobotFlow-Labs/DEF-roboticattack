.PHONY: backend-info dry-run bench cuda-setup cuda-smoke cuda-bench

backend-info:
	ANIMA_BACKEND=auto def-roboticattack backend-info

dry-run:
	ANIMA_BACKEND=auto def-roboticattack dry-run --batch-size 4 --height 224 --width 224

bench:
	python benchmarks/benchmark_patch_guard.py --backend auto --iterations 200 --batch-size 8

cuda-setup:
	bash scripts/cuda_server_setup.sh

cuda-smoke:
	bash scripts/cuda_smoke.sh

cuda-bench:
	bash scripts/cuda_benchmark.sh
