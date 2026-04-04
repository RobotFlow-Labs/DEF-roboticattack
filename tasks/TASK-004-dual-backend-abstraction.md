# TASK-004: Dual Backend Abstraction
## Status: DONE
## Assigned: build-agent
## Priority: P0
## Estimated: 3h

### Description
Create backend resolution layer supporting `ANIMA_BACKEND=auto|cuda|mlx|cpu`.

### Acceptance Criteria
- [x] Backend capability metadata returned at runtime
- [x] Explicit errors for unavailable forced backends
- [x] Auto mode prefers CUDA, then MLX, then CPU

### Dependencies
- Requires: TASK-003
