# Benchmarks & GPU tooling

## benchmarks/
Ollama serving benchmark results (from `~/tools/ollama-bench`, upstream: github.com/dalist1/ollama-bench).
One `.txt` per GPU / power-limit run.

## gpu-tooling/p2p/
- `p2p-kernel-wrapper.patch` — patch for NVIDIA open-gpu-kernel-modules enabling P2P on mixed-generation GPUs (applied to a fork of `aikitoria/open-gpu-kernel-modules`, branch state in `~/tools/nvidia-p2p-615.71.09`)
- `docker-compose.p2p-test.yaml` — vLLM compose variant used for P2P on/off testing

Upstream sources (not committed here, reference only):
- cuda-samples: github.com/NVIDIA/cuda-samples (local: `~/tools/cuda-samples`, `~/tools/cuda-samples-12.4`)
- gpu-burn: github.com/wilicc/gpu-burn (local: `~/tools/gpu-burn`)
- open-gpu-kernel-modules fork: `~/tools/nvidia-p2p-615.71.09`
