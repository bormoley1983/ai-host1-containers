docker network create llm-backends
docker network create llm-clients

llm-backends:
  LiteLLM
  Ollama
  vLLM
  llama.cpp

llm-clients:
  LiteLLM
  OpenWebUI
  ComfyUI
  call-analytics application container

Later:



                        llm-backends
                   ┌──────────────────────┐
                   │                      │
             Ollama                 llama.cpp
                   \                    /
                    \                  /
                     ──── LiteLLM ─────
                              │
                         llm-clients
                              │
              ┌───────────────┼────────────────┐
              │               │                │
          OpenWebUI     call-analytics    personal apps
              │
       media-generation
              │
           ComfyUI


llm-backends
  LiteLLM <-> Ollama/vLLM/llama.cpp

llm-clients
  LiteLLM <-> OpenWebUI/call-analytics/personal apps

media-generation
  OpenWebUI <-> ComfyUI


Compose examples
LiteLLM
services:
  litellm:
    container_name: litellm
    networks:
      - llm-backends
      - llm-clients

networks:
  llm-backends:
    external: true
  llm-clients:
    external: true
Ollama
services:
  ollama:
    container_name: ollama
    networks:
      - default
      - llm-backends

networks:
  llm-backends:
    external: true
OpenWebUI
services:
  open-webui:
    container_name: open-webui
    networks:
      - default
      - llm-clients
      - media-generation

networks:
  llm-clients:
    external: true
  media-generation:
    external: true
ComfyUI
services:
  comfyui:
    container_name: comfyui
    networks:
      - default
      - media-generation

networks:
  media-generation:
    external: true
Call analytics
services:
  app:
    networks:
      - default
      - llm-clients

  worker:
    networks:
      - default

networks:
  llm-clients:
    external: true
