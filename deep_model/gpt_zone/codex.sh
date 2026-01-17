#!/bin/bash
set -x

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

# docker compose down --remove-orphans --rmi local
# docker builder prune -af
# docker compose build --no-cache --pull

#cd docker && docker compose run --rm codex codex --dangerously-bypass-approvals-and-sandbox
#cd docker && docker compose run --rm codex_env codex --sandbox danger-full-access --ask-for-approval never
cd docker && docker compose run --rm codex_env codex 


# non interacitve
#docker compose run --rm -e CODEX_API_KEY="$CODEX_API_KEY" codex \
#  codex exec "Run the full test suite and fix failures until everything passes."

