#!/usr/bin/env bash
# Profile FlyDSL or CK warp-decode kernels with rocprof-compute.
#
# Usage:
#   bash profile.sh flydsl [--shapes SHAPE] [--batches B] [--no-roof]
#   bash profile.sh ck     [--shape SHAPE]  [--B B]       [--no-roof]
#
# Examples:
#   bash profile.sh flydsl --shapes deepseek-v3 --batches 1
#   bash profile.sh ck --shape deepseek-v3 --B 1
#   bash profile.sh flydsl --shapes qwen3next --batches 1 --no-roof
#
# rocprof-compute version: 3.6.0 (/opt/rocm/bin/rocprof-compute)
# Output directory: ./profile_out/<target>_<timestamp>/
#
# After profiling, view with:
#   rocprof-compute analyze -p ./profile_out/<dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCPROFCOMPUTE="/opt/rocm/bin/rocprof-compute"
OUTBASE="${SCRIPT_DIR}/profile_out"
mkdir -p "${OUTBASE}"

TARGET="${1:-}"
if [ -z "${TARGET}" ]; then
    echo "Usage: bash profile.sh [flydsl|ck] [options]" >&2
    exit 1
fi
shift

# Parse remaining args
SHAPES="deepseek-v3"
BATCHES="1"
ROOF="--roof-only"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shapes)  SHAPES="$2"; shift 2 ;;
        --shape)   SHAPES="$2"; shift 2 ;;
        --batches) BATCHES="$2"; shift 2 ;;
        --B)       BATCHES="$2"; shift 2 ;;
        --no-roof) ROOF=""; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTBASE}/${TARGET}_${TIMESTAMP}"
mkdir -p "${OUTDIR}"

echo "Profiling target=${TARGET} shapes=${SHAPES} batches=${BATCHES}"
echo "Output: ${OUTDIR}"
echo ""

case "${TARGET}" in
    flydsl)
        BENCH_CMD="python ${SCRIPT_DIR}/bench_flydsl_wd_bart.py \
            --shapes ${SHAPES} --batches ${BATCHES} --iters 10 --warmup 2"
        FLYDSL_RUNTIME_ENABLE_CACHE=0 \
        FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
        "${ROCPROFCOMPUTE}" profile \
            --name "wd_flydsl" \
            --output-directory "${OUTDIR}" \
            ${ROOF} \
            -- ${BENCH_CMD}
        ;;

    ck)
        CK_BENCH="${CK_BENCH:-/home/AMD/bartgips/code/rocm-libraries-wdec/bench_ck_warp_decode}"
        if [ ! -f "${CK_BENCH}" ]; then
            echo "ERROR: CK bench binary not found at ${CK_BENCH}." >&2
            echo "Run build_ck_bench.sh first, then set CK_BENCH env var." >&2
            exit 1
        fi
        BATCHES_CSV="${BATCHES// /,}"
        SHAPES_CSV="${SHAPES// /,}"
        CK_WD_SHAPES="${SHAPES_CSV}" \
        CK_WD_BATCHES="${BATCHES_CSV}" \
        CK_WD_ITERS="10" \
        CK_WD_COLD="2" \
        "${ROCPROFCOMPUTE}" profile \
            --name "wd_ck" \
            --output-directory "${OUTDIR}" \
            ${ROOF} \
            -- "${CK_BENCH}"
        ;;

    *)
        echo "Unknown target: ${TARGET}. Choose 'flydsl' or 'ck'." >&2
        exit 1
        ;;
esac

echo ""
echo "Profile data saved to: ${OUTDIR}"
echo ""
echo "To analyze (terminal):"
echo "  rocprof-compute analyze -p ${OUTDIR}"
echo ""
echo "To open GUI:"
echo "  rocprof-compute analyze -p ${OUTDIR} --gui"
