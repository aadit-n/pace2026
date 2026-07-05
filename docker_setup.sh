#!/usr/bin/env bash
set -euo pipefail

# Build helper for a fresh Debian 13.5 container.
# Run from the repository root:
#   ./docker_setup.sh

if [[ ! -f Makefile || ! -d src ]]; then
    echo "error: run this script from the repository root" >&2
    exit 1
fi

APT_GET="apt-get"
if [[ "${EUID}" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "error: not running as root and sudo is unavailable" >&2
        exit 1
    fi
    APT_GET="sudo apt-get"
fi

export DEBIAN_FRONTEND=noninteractive

${APT_GET} update
${APT_GET} install -y --no-install-recommends \
    ca-certificates \
    g++

rm -f pace_solver
g++ -O3 -march=native -flto -fno-plt -fno-rtti -fdevirtualize-speculatively \
    -Isrc -Isrc/io -Isrc/reductions \
    src/main.cpp -o pace_solver -pthread

echo
echo "Build complete: ./pace_solver"
echo "Example usage:"
echo "  ./pace_solver < input_instance > output_solution"
