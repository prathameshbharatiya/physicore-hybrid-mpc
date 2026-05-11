#!/usr/bin/env bash
# PhysiCore one-line installer
# Usage: curl -fsSL https://raw.githubusercontent.com/prathameshbharatiya/physicore-hybrid-mpc/main/install.sh | bash

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC}  $1"; }
fail() { echo -e "  ${RED}✗${NC}  $1"; exit 1; }
info() { echo -e "  ${CYAN}→${NC}  $1"; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║          PHYSICORE INSTALLER v1.3.0                  ║"
echo "║  Real-time physics adaptation. Any robot. 30 seconds.║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Python check
if ! command -v python3 &>/dev/null; then
    fail "Python 3.9+ is required. Install from https://python.org"
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    fail "Python 3.9+ required. Found: $PY_VERSION"
fi
ok "Python $PY_VERSION"

# Clone or update
INSTALL_DIR="$HOME/.physicore/src"
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing installation..."
    git -C "$INSTALL_DIR" pull --quiet
    ok "Updated"
else
    info "Cloning PhysiCore..."
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --quiet https://github.com/prathameshbharatiya/physicore-hybrid-mpc.git "$INSTALL_DIR"
    ok "Cloned to $INSTALL_DIR"
fi

# Install
info "Installing PhysiCore..."
python3 -m pip install --upgrade pip -q
python3 -m pip install -e "$INSTALL_DIR[all]" -q
ok "PhysiCore installed"

# Verify
if python3 -c "import physicore; print(physicore.__version__)" &>/dev/null; then
    VERSION=$(python3 -c "import physicore; print(physicore.__version__)")
    ok "PhysiCore $VERSION loaded"
else
    fail "Installation verification failed"
fi

# Registry
mkdir -p "$HOME/.physicore/registry" "$HOME/.physicore/plugins" "$HOME/.physicore/sessions"
ok "Data directories: ~/.physicore/"

echo ""
echo -e "${BOLD}${GREEN}Installation complete!${NC}"
echo ""
echo -e "  Start API server:  ${CYAN}physicore serve${NC}"
echo -e "  Check status:      ${CYAN}physicore status${NC}"
echo -e "  Run an example:    ${CYAN}physicore run balancing_bot_sim${NC}"
echo -e "  Open docs:         ${CYAN}physicore docs${NC}"
echo ""
