docker network ls | grep -E 'llm-backends|llm-clients'


KEY="sk-$(openssl rand -hex 32)"
printf 'LITELLM_MASTER_KEY=%s\n' "$KEY" > .env
chmod 600 .env

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
