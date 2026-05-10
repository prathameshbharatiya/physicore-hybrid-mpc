#!/usr/bin/env bash
# PhysiCore health check — hits the API and checks WebSocket port
# Usage: ./scripts/healthcheck.sh [host] [port]
# Returns exit code 0 on pass, 1 on fail

HOST="${1:-localhost}"
HTTP_PORT="${2:-8000}"
WS_PORT="${WS_PORT:-8765}"

PASS=0; FAIL=1
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

ALL_OK=true

# ── Check 1: HTTP health endpoint ─────────────────────────────────────────────
HTTP_RESULT=$(curl -sf --max-time 5 "http://${HOST}:${HTTP_PORT}/health" 2>&1 || echo "CURL_FAIL")
if echo "$HTTP_RESULT" | grep -q '"status".*"ok"'; then
    pass "HTTP /health → ok  (http://${HOST}:${HTTP_PORT}/health)"
else
    fail "HTTP /health failed: $HTTP_RESULT"
    ALL_OK=false
fi

# ── Check 2: /api/platforms responds ──────────────────────────────────────────
PLATFORMS=$(curl -sf --max-time 5 "http://${HOST}:${HTTP_PORT}/api/platforms" 2>&1 || echo "CURL_FAIL")
if echo "$PLATFORMS" | grep -q "quadrotor"; then
    pass "API /api/platforms → lists platforms"
else
    fail "API /api/platforms failed or missing 'quadrotor': $PLATFORMS"
    ALL_OK=false
fi

# ── Check 3: WebSocket bridge port ────────────────────────────────────────────
if command -v nc >/dev/null 2>&1; then
    if nc -z -w2 "$HOST" "$WS_PORT" 2>/dev/null; then
        pass "WebSocket bridge port ${WS_PORT} → open"
    else
        fail "WebSocket bridge port ${WS_PORT} not reachable (bridge not running?)"
        # Non-fatal — bridge is optional
    fi
else
    echo "[SKIP] nc not available — skipping WebSocket port check"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
if $ALL_OK; then
    pass "All checks passed."
    exit $PASS
else
    fail "One or more checks failed."
    exit $FAIL
fi
