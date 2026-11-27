# Llama.cpp Multi-GPU + Swift-1.5 Plan

**Date:** 2026-10-27
**Status:** Proposed v2 — no files modified yet
**Architecture decision:** GPU 1 = llamacpp coding (dedicated). GPU 0 = shared workhorse (ollama + ComfyUI + dev projects). Both routed through litellm.

## Current State

| Component | Detail |
|-----------|--------|
| GPUs | 2x RTX 3090 (24 GB each) |
| GPU 0 | ~266 MB used (ComfyUI python, idle) |
| GPU 1 | ~23.8 GB used by llamacpp (Qwen3.8-27B, 144K ctx, MTP) |
| CPU | Ryzen 9 9900X (12C/24T) |
| RAM | 29 GB total, ~18 GB available |
| llamacpp | `ghcr.io/ggml-org/llama.cpp:server-cuda`, port 8080, GPU 1 only |
| ollama | Running (no models loaded), `OLLAMA_SCHED_SPREAD=true`, `OLLAMA_MAX_LOADED_MODELS=2` |
| litellm | Port 4000, currently routes to ollama only (gemma4, qwen3.6) |

### Active llamacpp config (from `.env` + `docker-compose.qwen3_8_27B-MTP-latest.yml`)

```
Model:      Qwen3.8-27B-UD-Q4_K_XL.gguf (17.9 GB, 64 layers, hybrid SSM/full-attn)
GPU:        device_ids ["1"] only
Ctx:        147456 tokens
KV cache:   q8_0, flash-attn on, kv-unified, kv-offload
MTP:        draft-mtp, n-max=2, p-min=0.85
Parallel:   1 slot
Batch:      512 / ubatch 256
Threads:    10
```

### VRAM math (Qwen3.8-27B)

The model is hybrid: 48 SSM/linear-attention layers (no KV cache) + 16 full-attention
anchor layers (blk.3,7,11,...,63). KV per token = 4 KV-heads x 256 dim x 2 (K+V) x q8_0
= 32 KB/token across all full-attn layers.

| Ctx | KV cache | Weights + MTP | Total | Fits 24 GB? |
|-----|----------|---------------|-------|-------------|
| 65536 | 2.0 GB | 18.4 GB | 20.4 GB | Yes |
| 131072 | 4.0 GB | 18.4 GB | 22.4 GB | Yes (tight) |
| 147456 | 4.5 GB | 18.4 GB | 22.9 GB | Yes (very tight) |

The current 144K config is at the edge of a single 24 GB GPU. Multi-GPU split halves
per-GPU load and unlocks headroom for parallel slots or higher ctx.

---

## Swift-1.5 Capabilities

**File:** `Swift-1.5-Qwen3.8-27B-Uncensored-VAL-APEX-I-MiniPlus-V3.gguf` (14.16 GB)
**Source:** IsValorum APEX-I MiniPlus V3 quantization of the Swift-1.5 Qwen3.8 27B
Uncensored MTP base model.

### What it is
- Uncensored / abliterated variant of Qwen3.8 27B (refusal behavior removed)
- "Swift" branding = fine-tuned for speed and directness in agentic workflows
- APEX-I MiniPlus V3: tensor-by-tensor quantization, Q5-class fidelity at ~14 GB
- WikiText-2 PPL: +5.70% vs GSQ-RCO IQ3_S baseline (per README)
- Same architecture as Qwen3.8-27B: 64 layers, hybrid SSM/full-attn, MTP support

### What it has vs base Qwen3.8
- **No refusal filtering** — will comply with requests the base model would decline
- **MTP companion** — supports speculative decoding (same `--spec-type draft-mtp`)
- **Smaller file** (14.16 GB vs 17.92 GB) — more VRAM headroom for KV cache
- **Same tool-calling capability** — uses the same Qwen chat template with function
  calling support
- **Vision** — the `mmproj-Q8_0.gguf` (614 MB) projector is compatible

### What it lacks vs base Qwen3.8
- Lower quantization fidelity than UD-Q4_K_XL (MiniPlus V3 uses more aggressive
  per-tensor quant; PPL delta is higher)
- Uncensored behavior may produce less cautious code in edge cases (no safety rails)
- No independent benchmark data for agentic coding specifically (the README benchmarks
  are for the Cyber-Tiel-Coder MoE line, not this dense model)

### VRAM on single GPU

| Ctx | KV cache | Weights + MTP | Total | Fits 24 GB? |
|-----|----------|---------------|-------|-------------|
| 65536 | 2.0 GB | 14.7 GB | 16.7 GB | Yes (comfortable) |
| 131072 | 4.0 GB | 14.7 GB | 18.7 GB | Yes |
| 147456 | 4.5 GB | 14.7 GB | 19.2 GB | Yes |

Swift-1.5 has ~3.7 GB more headroom than Qwen3.8 at the same ctx — enough for
`--parallel 2` slots on a single GPU.

### Testing plan (no file changes)
1. Run Swift-1.5 on GPU 0 with the existing compose file pattern, port 8081
2. Point litellm at it as a separate model entry
3. A/B test: same agentic coding prompt through Qwen3.8 (GPU 1) vs Swift-1.5 (GPU 0)
4. Measure: tok/s, tool-call accuracy, refusal rate, code correctness

---

## Multi-GPU Strategy (revised)

**Architecture:** GPU 1 is dedicated to llamacpp coding. GPU 0 is a shared workhorse
for ollama (wide model range), ComfyUI, and dev projects (call analytics, video
analytics). Both backends route through litellm.

### Why not split Qwen3.8 across both GPUs?

The original Option A (split one model across two GPUs) is rejected because GPU 0
must remain available for the wide range of other workloads. Splitting would lock
both GPUs to a single model and eliminate the multi-use capability that is the
primary reason for having two GPUs.

### What multi-GPU DOES give us here

1. **Parallelism, not speed:** llamacpp coding on GPU 1 runs concurrently with
   ollama/ComfyUI/dev work on GPU 0. No contention, no time-slicing.
2. **Model diversity:** GPU 0 hosts a rotating cast of models via ollama
   (ornith-1.5:9b for sub-agents, qwen3.5:4b for fast tasks, gemma4 for general
   work, plus any model pulled for reference). GPU 1 stays locked to the best
   coding model (Qwen3.8 or Swift-1.5).
3. **Multi-agent routing:** litellm routes by model_name — the harness picks
   Qwen3.8 for deep reasoning and ornith-1.5:9b (or similar) for fast tool-calling,
   search, and summarization sub-tasks.

### GPU 0 workload budget (24 GB)

| Workload | VRAM (typical) | Notes |
|----------|---------------|-------|
| ComfyUI | ~2-4 GB | Currently 266 MB idle; spikes during generation |
| ollama model (ornith-1.5:9b, 64K ctx) | ~6.2 GB | A3B active, fast |
| ollama model (qwen3.5:4b, 32K ctx) | ~3.7 GB | Fast utility model |
| Dev project (call analytics / video) | variable | Whisper ~1-2 GB, DeepStream ~4-8 GB |
| **Concurrent worst case** | **~12-16 GB** | Fits with headroom |

ollama's `OLLAMA_SCHED_SPREAD=true` already distributes models across GPUs, but
since GPU 1 is pinned to llamacpp via `device_ids`, ollama effectively only sees
GPU 0 (llamacpp's container reservation doesn't block other containers from seeing
the GPU — see "GPU isolation" note below).

### GPU isolation note

Docker's `deploy.resources.reservations.devices` does NOT hard-exclude a GPU from
other processes — it only controls which GPUs are visible to that container. The
llamacpp container sees GPU 1 (via `device_ids: ["1"]`), but ollama and ComfyUI
containers will see BOTH GPUs unless explicitly restricted.

To keep ollama off GPU 1, set in the ollama container env:
```
CUDA_VISIBLE_DEVICES=0
```
This is a soft restriction (ollama respects it) but prevents accidental model
placement on the coding GPU. Same for ComfyUI and any dev project containers.

llamacpp's `.env` already has `CUDA_VISIBLE_DEVICES=1` which keeps it on GPU 1.
The compose file's `device_ids: ["1"]` is redundant with this but provides a
harder guarantee at the container level. Keep both.

---

## Recommended Plan (revised)

**Phase 1: Test Swift-1.5 on GPU 0 (immediate, no changes to existing files)**

Create a new compose file `docker-compose.swift1.5-test.yml` (draft in plan below).
Run it on GPU 0 port 8081, verify it loads and responds, run a coding prompt through
it. This is a temporary test container — stop it when done to free GPU 0 for ollama.

**Phase 2: litellm consolidation (move to this host)**

Move litellm from 192.168.100.50 to this host so it can reach both llamacpp
(localhost:8080) and ollama (localhost:11434) over the local docker network without
cross-host latency. OpenWebUI stays on .50 and points at this host's litellm
(192.168.100.X:4000).

**Why move:**
- litellm → llamacpp and litellm → ollama become localhost calls (sub-ms vs LAN)
- Single config file manages all models in one place
- No cross-host network failure mode between router and backends
- OpenWebUI on .50 still works — it just calls litellm over LAN (one hop, not two)

**What to do:**
1. Copy `/home/admaccess/litellm/` to this host (config.yaml + .env + docker-compose.yaml)
2. Update `api_base` values: `http://ollama:11434` stays (same docker network),
   add `http://llamacpp:8080/v1` for the OpenAI-compatible endpoint
3. Ensure both containers share a docker network (litellm's compose uses external
   networks `llm-backends` and `llm-clients` — create or join them)
4. Update OpenWebUI on .50 to point at new litellm address
5. Stop litellm on .50

**Phase 3: Add llamacpp + sub-agent entries to litellm config**

The current `config.yaml` only has ollama entries (gemma4, qwen3.6). Add:
- Qwen3.8 via llamacpp OpenAI endpoint (the primary coding model)
- Swift-1.5 if it survives Phase 1 testing
- ornith-1.5:9b via ollama (sub-agent for fast tool-calling)
- qwen3.5:4b via ollama (utility/fast tasks)

(See litellm config draft below.)

**Phase 4: Multi-agent harness routing**

Configure the harness to route by task type:
- Deep reasoning / complex code generation → Qwen3.8 (GPU 1, llamacpp)
- Fast tool-calling / file search / summarization → ornith-1.5:9b (GPU 0, ollama)
- General chat / quick questions → qwen3.5:4b or gemma4 (GPU 0, ollama)

This is the multi-agent win: the big model thinks, the small model acts.

```yaml
  # Main agentic coding model (Qwen3.8, GPU 1 via llamacpp)
  - model_name: local-qwen3.8:27B-144K
    litellm_params:
      model: openai/local-qwen3.8:27B-144K
      api_base: http://llamacpp:8080/v1
      api_key: not-needed

      temperature: 0.7
      top_p: 0.8
      top_k: 20
      min_p: 0.0
      presence_penalty: 1.5
      repeat_penalty: 1.0

      timeout: 1200

    model_info:
      max_input_tokens: 139264   # 147456 - 8192 reserve
      max_output_tokens: 16384
      supports_function_calling: true
      supports_reasoning: true

  # Swift-1.5 uncensored (GPU 0 via llamacpp, if adopted after testing)
  - model_name: local-swift1.5:27B-128K
    litellm_params:
      model: openai/local-swift1.5:27B-128K
      api_base: http://swift15-test:8081/v1
      api_key: not-needed

      temperature: 0.7
      top_p: 0.8
      top_k: 20
      min_p: 0.0

      timeout: 1200

    model_info:
      max_input_tokens: 122880
      max_output_tokens: 16384
      supports_function_calling: true
      supports_reasoning: true

  # Sub-agent (ornith-1.5:9b via ollama, GPU 0) — fast tool-calling
  - model_name: local-ornith:9B-64K
    litellm_params:
      model: ollama_chat/ornith-1.5:9b
      api_base: http://ollama:11434

      num_ctx: 65536
      num_predict: 4096

      temperature: 0.6
      top_p: 0.95
      top_k: 20

      think: true
      timeout: 600

    model_info:
      max_input_tokens: 61440
      max_output_tokens: 4096
      supports_function_calling: true
      supports_reasoning: true

  # Utility (qwen3.5:4b via ollama, GPU 0) — fast general tasks
  - model_name: local-qwen3.5:4B-32K
    litellm_params:
      model: ollama_chat/qwen3.5:4b
      api_base: http://ollama:11434

      num_ctx: 32768
      num_predict: 4096

      temperature: 0.7
      top_p: 0.9
      top_k: 20

      timeout: 300

    model_info:
      max_input_tokens: 28672
      max_output_tokens: 4096
      supports_function_calling: true
      supports_reasoning: false
```

Note: litellm currently has no entry for the llamacpp endpoint at all — it only routes
to ollama. The Qwen3.8 model is accessible to the harness via litellm's OpenAI-compatible
proxy, but the config.yaml doesn't declare it. This means either:
1. The harness talks to llamacpp directly (bypassing litellm), or
2. There's a missing entry that should be added

Moving litellm to this host (Phase 2) resolves this cleanly — add the llamacpp entry
and the harness routes everything through one endpoint.

---

## .env Changes Needed

**llamacpp `.env` (GPU 1, no changes):**
```
CUDA_VISIBLE_DEVICES=1          # already set, keep as-is
```
No changes needed for the primary coding model. The current config is correct for
single-GPU operation on GPU 1.

**ollama container env (GPU 0 isolation):**
```
CUDA_VISIBLE_DEVICES=0          # ADD — keeps ollama off the coding GPU
OLLAMA_SCHED_SPREAD=true        # already set, keep
OLLAMA_MAX_LOADED_MODELS=2      # already set, keep
```

**ComfyUI + dev project containers:**
```
CUDA_VISIBLE_DEVICES=0          # ADD to each — pins them to GPU 0
```

**litellm (new on this host):**
No GPU env needed — litellm is CPU-only, it just proxies HTTP.

---

## Docker Network Topology (after consolidation)

```
This host (192.168.100.X)
├── docker network: llm-backends (external)
│   ├── llamacpp        (GPU 1, port 8080)  ← coding model
│   ├── ollama          (GPU 0, port 11434) ← utility + sub-agent models
│   └── swift15-test    (GPU 0, port 8081)  ← temporary test container
│
├── docker network: llm-clients (external)
│   ├── litellm         (port 4000)         ← router, CPU-only
│   └── (harness containers join here)
│
Host port mapping:
  8080  → llamacpp  (also accessible directly if needed)
  11434 → ollama    (also accessible directly)
  4000  → litellm   (OpenWebUI on .50 points here)

Host .50 (192.168.100.50)
├── OpenWebUI → points at 192.168.100.X:4000 (litellm on this host)
```

All three LLM backends share the `llm-backends` network so litellm can reach them by
container name. The harness joins `llm-clients` to reach litellm. If the harness
runs on this host, it can also reach llamacpp/ollama directly via localhost ports.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU 0 VRAM contention (ollama + ComfyUI + dev) | OOM on one workload | `OLLAMA_MAX_LOADED_MODELS=2` caps ollama; stop ComfyUI during heavy inference; monitor with `nvidia-smi -l 5` |
| Swift-1.5 quantization quality | Lower code correctness than Q4_K_XL | A/B test on real coding tasks before adopting; keep Qwen3.8 as primary |
| ornith-1.5:9b too weak for sub-agent | Poor tool-calling accuracy | Fall back to qwen3.5:4b or run Qwen3.6-27B as sub-agent on GPU 0 (fits at 32K ctx) |
| litellm move breaks OpenWebUI | UI down during migration | Update OpenWebUI base URL before stopping old litellm; test from .50 first |
| ollama accidentally loads model on GPU 1 | VRAM contention with coding model | `CUDA_VISIBLE_DEVICES=0` in ollama container env (soft but effective) |
| Swift-1.5 test container blocks GPU 0 | Can't run ollama/ComfyUI during test | Test is temporary — stop the container when done; or run test during a window when GPU 0 is idle |

---

## Open Questions

1. **Harness routing:** Does the harness support per-sub-task model routing (different
   model for tool-calling vs reasoning), or does it use a single model_name for all
   agent roles? If single-model, the multi-agent win is limited to what litellm's
   fallback chain can do.

2. **litellm move timing:** Is there a maintenance window for moving litellm from .50
   to this host? OpenWebUI will be briefly unavailable during the cutover.

3. **ComfyUI persistence:** Is ComfyUI on GPU 0 a persistent workload (always running)
   or on-demand? If persistent, it reserves ~2-4 GB VRAM continuously and reduces
   headroom for ollama models.

4. **Swift-1.5 test window:** When is GPU 0 free for the Swift-1.5 test? The test
   container needs ~19 GB VRAM at 128K ctx, which conflicts with any loaded ollama
   model or active ComfyUI session.
