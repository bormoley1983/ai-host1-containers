# ai-host1-containers

Single source of truth for the AI host's infra configs. Service dirs here are the **live** compose projects — containers run from this repo, and it doubles as the git backup.

## Layout

| Dir | What it holds | Status |
|---|---|---|
| `ollama/` | Ollama compose + model update script + network notes | live (container runs from here) |
| `comfyui/` | ComfyUI compose files, custom nodes, user scripts, settings; runtime state in `input/ output/ cache/ user/` (git-ignored) | live (container runs from here) |
| `litellm/` | LiteLLM proxy configs (model_list) + compose + `litellm-pgvector` build context. Secrets in `.env` / `.env.pgvector` (git-ignored) | **pending migration** (still at `~/litellm`) |
| `vllm/` | vLLM compose files (incl. P2P test variant) | **pending migration** (still at `~/vllm`; not running) |
| `llamacpp/` | llama.cpp server env templates (per-context-size), compose variants, multi-GPU plan; `.venv` git-ignored | **pending migration** (still at `~/llamacpp`; serves the active agent model — migrate last) |
| `benchmarks/` | Ollama serving benchmark results per GPU / power limit (from `~/ollama-bench`) | backup copy |
| `gpu-tooling/p2p/` | P2P kernel patch + vLLM P2P test compose (see `benchmarks-and-tooling.md`) | backup copy |

## Running services

```bash
cd ~/ai-host1-containers/<service>
docker compose up -d        # start
docker compose down         # stop
```

Models live on `/models/*` (outside this repo). ComfyUI runtime state lives in `comfyui/{input,output,cache,user}`.

## Secrets

Real credentials (`litellm/.env`, `litellm/.env.pgvector`) are git-ignored but kept **in the service dir** so compose finds them. Configs reference them via `os.environ/...`. The backup script (`~/bin/backup-ai-infra.sh`) scans staged content for credential patterns and aborts the push if one appears.

## Backing up

```bash
~/bin/backup-ai-infra.sh    # commit + push any config changes (idempotent)
```
