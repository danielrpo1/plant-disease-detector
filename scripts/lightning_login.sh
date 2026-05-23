#!/usr/bin/env bash
# Carga credenciales desde .env y ejecuta lightning login
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PATH="${HOME}/Library/Python/3.9/bin:${PATH}"

if [[ -z "${LIGHTNING_USER_ID:-}" || -z "${LIGHTNING_API_KEY:-}" ]]; then
  echo "Falta LIGHTNING_USER_ID o LIGHTNING_API_KEY en .env"
  exit 1
fi

lightning login
echo "OK — Lightning AI autenticado."
