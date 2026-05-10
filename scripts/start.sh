#!/usr/bin/env bash
# PhysiCore — single-command dev launcher
# Usage: ./scripts/start.sh
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; AMBER='\033[0;33m'; NC='\033[0m'

info()  { echo -e "${GREEN}[START]${NC} $*"; }
warn()  { echo -e "${AMBER}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Version checks ─────────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || echo "")
[ -z "$PYTHON" ] && error "Python not found. Install Python >= 3.10."

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python >= 3.10 required (found $PY_VER)"
fi
info "Python $PY_VER ✓"

NODE=$(command -v node || echo "")
[ -z "$NODE" ] && error "Node.js not found. Install Node >= 18."
NODE_VER=$(node --version | sed 's/v//')
NODE_MAJOR=$(echo "$NODE_VER" | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 18 ]; then
    error "Node >= 18 required (found v$NODE_VER)"
fi
info "Node v$NODE_VER ✓"

# ── Install frontend deps if missing ──────────────────────────────────────────
if [ ! -d "node_modules" ]; then
    info "Installing frontend dependencies..."
    npm install
fi

# ── Install PhysiCore Python package if not installed ─────────────────────────
if ! "$PYTHON" -c "import physicore" 2>/dev/null; then
    info "Installing PhysiCore Python package..."
    "$PYTHON" -m pip install -e . --quiet
fi

# ── Start both servers ────────────────────────────────────────────────────────
info "Starting PhysiCore backend on :8000 and Vite dev server on :5173..."
info "Press Ctrl+C to stop both."

# Cleanup on exit
cleanup() {
    echo ""
    info "Shutting down..."
    [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    wait 2>/dev/null
    info "Done."
}
trap cleanup INT TERM

# Start backend
"$PYTHON" -m uvicorn physicore.api.server:app \
    --host 0.0.0.0 --port 8000 --reload \
    2>&1 | sed "s/^/${AMBER}[BACKEND]${NC} /" &
BACKEND_PID=$!

# Give backend a moment to start
sleep 1

# Start frontend Vite dev server
npm run dev 2>&1 | sed "s/^/${GREEN}[FRONTEND]${NC} /" &
FRONTEND_PID=$!

# Wait for either to exit
wait -n 2>/dev/null || wait
