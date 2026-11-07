#!/usr/bin/env bash
set -euo pipefail

# Download the UniCo prompt files into ./data/unico/. Run from the repository root.
# Extra arguments are forwarded, for example: --splits test train

python3 -m id_eval.download_unico_data "$@"
