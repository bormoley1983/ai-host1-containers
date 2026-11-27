# ai-host1-containers

Config backup for the AI host (LLM serving stack). Per-service docker-compose files, model configs, and env templates.

## Layout

| Dir | What it holds |
|---|---|
| `ollama/` | Ollama compose + model update script + network notes |
| `litellm/` | LiteLLM proxy configs (model_list) + compose. Secrets via `.env` (not committed) |
| `vllm/` | vLLM compose files (incl. P2P test variant) |
| `llamacpp/` | llama.cpp server env templates (per-context-size), compose variants, multi-GPU plan |

## Secrets

`.env` files with real credentials are **not** committed (see `.gitignore`).
Configs reference them via `os.environ/...`. To run: copy a service's env template to `.env` and fill in values.
