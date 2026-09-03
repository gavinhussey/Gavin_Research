#!/bin/bash
# Wrapper invoked by the nightly launchd job (see launchd/*.plist.template
# in this directory). Never contains a credential itself -- sources an
# untracked, chmod-600 secrets file created once by hand at
# ~/.atlas-quant/secrets.env (SEC_EDGAR_USER_AGENT), never committed to
# this repo. See acquire_data_if_entry_eve.py for what this actually does.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SECRETS_FILE="$HOME/.atlas-quant/secrets.env"

if [ -f "$SECRETS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
fi

cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/live/multi_factor_ranking_ml/acquire_data_if_entry_eve.py"
