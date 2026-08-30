#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TONGJI_PYTHON:-/home/u/geocoding/tongji_sbas/.venv/bin/python}"

"${PYTHON_BIN}" "${ROOT}/code/run_iterative_triangle_adjustment.py" --config "${ROOT}/config.json" "$@"
