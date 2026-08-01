#!/bin/bash
# Wrapper invoked by the daily launchd job (see launchd/*.plist.template
# in this directory). Never contains a credential itself -- sources an
# untracked, chmod-600 secrets file created once by hand at
# ~/.atlas-quant/secrets.env (ALPACA_API_KEY, ALPACA_API_SECRET), never
# committed to this repo. See run_paper_trade.py for what this actually
# does -- this is the real, order-submitting path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SECRETS_FILE="$HOME/.atlas-quant/secrets.env"

if [ -f "$SECRETS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
fi

cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/live/filing_momentum_ml/run_paper_trade.py"
