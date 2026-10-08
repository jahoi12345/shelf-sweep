#!/usr/bin/env bash
# Resumable, pinned setup. --minimal installs only the core app + Spanish/English RapidOCR.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/setup_local_models.py "$@"
