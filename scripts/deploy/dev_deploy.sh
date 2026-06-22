#!/usr/bin/env bash
# Deploy the current dev checkout: sync deps, migrate, restart Streamlit.
# Safe to run by hand as streamlituser (operates on the already-updated tree).
set -euo pipefail

REPO_DIR="/home/streamlituser/thrive_ui"
UV="/home/streamlituser/.local/bin/uv"
SYSTEMCTL="/usr/bin/systemctl"

export PATH="/home/streamlituser/.local/bin:$PATH"  # systemd's env lacks ~/.local/bin
cd "$REPO_DIR"

echo "==> uv sync"
"$UV" sync

echo "==> alembic upgrade head"
"$UV" run alembic upgrade head

echo "==> restart streamlit"
sudo "$SYSTEMCTL" restart streamlit

echo "==> verify active"
"$SYSTEMCTL" is-active streamlit   # is-active needs no root; non-zero exit fails the deploy
echo "==> deploy OK"
