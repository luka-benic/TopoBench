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
#   MIN_FREE_MB=<n>      warn if the chosen GPU has less free memory (default 4000)
#   OPENQDC_CACHE_DIR    where OpenQDC data is stored (default: <repo>/datasets/openqdc)
#
# Before training, the dataset is checked and anything missing is done first:
# download the OpenQDC data, then preprocess it (TopoBench graphs or MACE .xyz
# files). The same step is available alone: scripts/openqdc/prepare_data.py.
#
# Several jobs can run at once (different GPUs, models or energy columns): data
# conversion, downloads, splits and .xyz export are protected by file locks.
# MACE runs get a default name from their data and settings; two runs with
# identical settings refuse to share a folder (use --name for the second one).
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
    # nvidia-smi numbers the GPUs by PCI bus; make CUDA use the same numbering,
    # so the card checked below is the card used.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    MACE_DEVICE="cuda"
    TRAINER="gpu"
    THREADS="${THREADS:-1}"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    if command -v nvidia-smi > /dev/null; then
        # Fail now, not after the data has been prepared, if the GPU is missing.
        if ! nvidia-smi -i "$DEVICE" > /dev/null 2>&1; then
            echo "GPU $DEVICE does not exist (see nvidia-smi)."
            exit 1
        fi
        nvidia-smi -i "$DEVICE" --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
        FREE_MB=$(nvidia-smi -i "$DEVICE" --query-gpu=memory.free --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ' || true)
        MIN_FREE_MB="${MIN_FREE_MB:-4000}"
        if [[ "$FREE_MB" =~ ^[0-9]+$ ]] && [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
            echo "WARNING: GPU $DEVICE has only $FREE_MB MiB free (less than $MIN_FREE_MB MiB), so the job may run out of memory."
            echo "         Pick a less used GPU (see nvidia-smi) or set MIN_FREE_MB to silence this warning."
        fi
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
