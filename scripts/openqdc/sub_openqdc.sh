#!/bin/bash
#
# Submission script: train MACE or TopoBench on an OpenQDC dataset, on one GPU
# or on the CPU. Uses the repo's uv environment (<repo>/.venv).
#
# Usage:
#   sub_openqdc.sh <mace|topobench> <gpu number|cpu> [arguments of the training script]
#
# Everything after the device is passed on to scripts/openqdc/train_mace.py or
# scripts/openqdc/train_topobench.py (see their --help). Arguments after a
# lone "--" go to mace_run_train / Hydra.
#
# Energy column (which of the dataset's energies to train on, by index or name):
#   - default: the energy_method set in configs/dataset/graph/openqdc_<name>.yaml
#   - ENERGY_METHOD=3 ./sub_openqdc.sh ...        (environment variable)
#   - ./sub_openqdc.sh ... --energy-method 3      (flag; wins over ENERGY_METHOD)
#   The list of columns is at the top of each dataset yaml.
#
# Examples:
#   ./sub_openqdc.sh mace 1 --dataset QM9 --max-samples 20000 --epochs 50
#   ENERGY_METHOD=3 ./sub_openqdc.sh topobench 1 --dataset QM9 --max-samples 20000
#   ./sub_openqdc.sh mace 1 --dataset QM9 --energy-method "B3LYP:def2-svp"
#   ./sub_openqdc.sh topobench 1 --dataset Alchemy --model graph/gcn --max-samples 20000
#   ./sub_openqdc.sh topobench cpu --dataset QM9 --model cell/cwn --max-samples 2000 --epochs 5
#   ./sub_openqdc.sh mace cpu --dataset QM9 --max-samples 2000 --epochs 5 -- --restart_latest
#
# Optional environment variables:
#   ENERGY_METHOD=<index or name>   energy column to train on (see above)
#   THREADS=<n>          CPU threads (default: 1 on a GPU run, 8 on a CPU run)
#   OPENQDC_CACHE_DIR    where OpenQDC data is stored (default: <repo>/datasets/openqdc)
#
# Run in the background with:  nohup ./sub_openqdc.sh mace 1 --dataset QM9 > run.log 2>&1 &

set -e

usage() {
    echo "Usage: $0 <mace|topobench> <gpu number|cpu> [arguments of the training script]"
    echo "  Must set the number of the CUDA device, or 'cpu'. Use nvidia-smi to get the numbers of available devices!"
    echo "  Example: $0 mace 1 --dataset QM9 --max-samples 20000 --epochs 50"
}

MODEL="${1:-}"
DEVICE="${2:-}"
if [ -z "$MODEL" ] || [ -z "$DEVICE" ]; then
    usage
    exit 1
fi
shift 2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Python of the uv environment not found: $PYTHON"
    exit 1
fi

case "$MODEL" in
    mace | topobench) ;;
    *)
        echo "First argument must be 'mace' or 'topobench', got '$MODEL'."
        usage
        exit 1
        ;;
esac

if [ "$DEVICE" = "cpu" ] || [ "$DEVICE" = "CPU" ]; then
    # Hide all GPUs so nothing can end up on one by accident.
    export CUDA_VISIBLE_DEVICES=""
    MACE_DEVICE="cpu"
    TRAINER="cpu"
    THREADS="${THREADS:-8}"
elif [[ "$DEVICE" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    MACE_DEVICE="cuda"
    TRAINER="gpu"
    THREADS="${THREADS:-1}"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    if command -v nvidia-smi > /dev/null; then
        nvidia-smi -i "$DEVICE" --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
    fi
else
    echo "Device must be one GPU number (e.g. 1) or 'cpu', got '$DEVICE'."
    usage
    exit 1
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

# An energy column given as an environment variable is passed before "$@", so an
# explicit --energy-method flag after the device overrides it.
ENERGY_ARGS=()
if [ -n "${ENERGY_METHOD:-}" ]; then
    ENERGY_ARGS=(--energy-method "$ENERGY_METHOD")
fi

cd "$REPO"
echo "Running $MODEL on $DEVICE with $THREADS thread(s); python: $PYTHON"

if [ "$MODEL" = "mace" ]; then
    "$PYTHON" scripts/openqdc/train_mace.py --device "$MACE_DEVICE" "${ENERGY_ARGS[@]}" "$@"
else
    "$PYTHON" scripts/openqdc/train_topobench.py "${ENERGY_ARGS[@]}" "$@" -- "trainer=$TRAINER"
fi

echo "Done."
