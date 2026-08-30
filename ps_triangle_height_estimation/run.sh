#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TONGJI_PYTHON:-/home/u/geocoding/tongji_sbas/.venv/bin/python}"

"${PYTHON_BIN}" "${ROOT}/code/map_ps_to_building_surfaces.py" --config "${ROOT}/config.json" "$@"
