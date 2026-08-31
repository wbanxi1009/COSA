#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TTA_METHOD="PETSA"
exec bash "${SCRIPT_DIR}/tta.sh"
