#!/usr/bin/env bash
#
# build_host.sh -- regenerate every artifact of an example on the host, in the right order.
#
#   scripts/build_host.sh <example_dir> [--skip-test] [-v]
#
#   --skip-test   skip the end-to-end quantized test (slow; informational only)
#   -v            show the full output of every step (it is always written to the logs)
#
# Run it inside the Vitis AI container with the conda environment active:
#   conda activate vitis-ai-pytorch
#
# Everything goes to build/<example>/, which is deleted and rebuilt on every run, so
# no artifact can survive from an older state. build/<example>/board/ is exactly what
# is copied to the board (see scripts/deploy_board.sh). Any failing step stops the build.
#
# Parameters of an example (model file, weights, data, ...) are read from <example_dir>/bridge.conf.
# The compiler architecture can be changed with the environment variable BRIDGE_ARCH.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() { echo "[build] error: $*" >&2; exit 1; }

# --! arguments
SKIP_TEST=0
VERBOSE=0
EXAMPLE=""
for arg in "$@"; do
    case "$arg" in
        --skip-test) SKIP_TEST=1 ;;
        -v)          VERBOSE=1 ;;
        -h|--help)   sed -n '2,19p' "$0"; exit 0 ;;
        -*)          die "unknown option $arg" ;;
        *)           EXAMPLE="$arg" ;;
    esac
done
[ -n "$EXAMPLE" ] || die "usage: $0 <example_dir> [--skip-test] [-v]"
[ -d "$EXAMPLE" ] || die "example directory $EXAMPLE does not exist"
EX="$(cd "$EXAMPLE" && pwd -P)"
[ -f "$EX/bridge.conf" ] || die "$EX/bridge.conf does not exist"

# --! parameters of the example
# shellcheck disable=SC1091
. "$EX/bridge.conf"
for v in MODEL CONFIG PIPELINE WEIGHTS DATA LABELS TRAJ_NSAMPLE; do
    [ -n "${!v:-}" ] || die "$v is not set in $EX/bridge.conf"
done
for v in MODEL CONFIG PIPELINE WEIGHTS DATA LABELS; do
    path="$EX/${!v}"
    [ -f "$path" ] || die "$v: file $path does not exist"
    printf -v "$v" '%s' "$path"
done

NAME="$(basename "$EX")"
OUT="$ROOT/build/$NAME"
HOST="$ROOT/src/host"
ARCH="${BRIDGE_ARCH:-/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json}"
export PYTHONPATH="$ROOT/src/common:$ROOT/src/board:$ROOT/src/host${PYTHONPATH:+:$PYTHONPATH}"

# --! environment
python3 -c "import torch, pytorch_nndct" 2>/dev/null \
    || die "torch or pytorch_nndct cannot be imported; run inside the Vitis AI container after 'conda activate vitis-ai-pytorch'"
command -v vai_c_xir >/dev/null 2>&1 || die "vai_c_xir not found in PATH"
[ -f "$ARCH" ] || die "architecture file $ARCH does not exist (set BRIDGE_ARCH)"

# --! a fresh output directory: nothing may survive from an earlier build
rm -rf "$OUT"
mkdir -p "$OUT/logs" "$OUT/golden" "$OUT/compiled"

# --! helpers: every step has its own log; the console shows only the lines that carry results
STEP_NO=0
STEP=""
LOG=""

begin() {
    STEP_NO=$((STEP_NO + 1))
    STEP="$1"
    LOG="$OUT/logs/$(printf '%02d' "$STEP_NO")_$1.log"
    : > "$LOG"
    echo
    echo "==== step $STEP_NO: $2"
}

show() {
    if [ "$VERBOSE" = 1 ]; then
        cat
    else
        grep -E '^\[(note|warn|error|result|check|accuracy|alignment|build)' || true
    fi
}

run() {
    echo "+ $*" >> "$LOG"
    "$@" 2>&1 | tee -a "$LOG" | show
}

trap 'echo >&2; echo "[build] FAILED in step $STEP_NO ($STEP); log: $LOG" >&2' ERR

plan_stages() {
    python3 - "$OUT/model_plan.json" << 'EOF'
import json, sys
plan = json.load(open(sys.argv[1]))
print(' '.join(n for n, t in plan['trace'].items() if t['ncalls'] > 0))
EOF
}

stage_class() {
    python3 - "$OUT/model_plan.json" "$1" << 'EOF'
import json, sys
print(json.load(open(sys.argv[1]))['stages'][sys.argv[2]]['class'])
EOF
}

COMMON=(--weights "$WEIGHTS" --data "$DATA" --traj-nsample "$TRAJ_NSAMPLE" --out-quant "$OUT/quantize_result")

echo "[build] example $NAME; output $OUT"

# ---------------------------------------------------------------------------------------
begin analyze "analyze the model: structure, stages and the row shapes they see"
run python3 "$HOST/analyze_model.py" "$MODEL" "$CONFIG" "${COMMON[@]}" --out-plan "$OUT/model_plan.json"

STAGES="$(plan_stages)"
[ -n "$STAGES" ] || die "the plan lists no stage that runs"
echo "[build] stages: $STAGES"

# ---------------------------------------------------------------------------------------
begin calibrate "calibrate every stage (one process per stage: the quantizer keeps global state)"
for s in $STAGES; do
    run python3 "$HOST/calib_quant.py" "$MODEL" "$CONFIG" "$s" "${COMMON[@]}"
done

# ---------------------------------------------------------------------------------------
if [ "$SKIP_TEST" = 0 ]; then
    begin test "test the quantized composite end to end (float vs quantized, accuracy, label alignment)"
    run python3 "$HOST/test_quant.py" "$MODEL" "$CONFIG" "${COMMON[@]}" --labels "$LABELS" --check-alignment
fi

# ---------------------------------------------------------------------------------------
begin export "export every quantized stage as an xmodel (one process per stage)"
for s in $STAGES; do
    run python3 "$HOST/export_quant.py" "$MODEL" "$CONFIG" "$s" "${COMMON[@]}"
    cls="$(stage_class "$s")"
    [ -f "$OUT/quantize_result/$s/${cls}_int.xmodel" ] \
        || die "exporting stage $s did not produce $OUT/quantize_result/$s/${cls}_int.xmodel"
done

# ---------------------------------------------------------------------------------------
begin compile "compile every stage for the DPU"
for s in $STAGES; do
    cls="$(stage_class "$s")"
    mkdir -p "$OUT/compiled/$s"
    run vai_c_xir -x "$OUT/quantize_result/$s/${cls}_int.xmodel" -a "$ARCH" -o "$OUT/compiled/$s" -n "$s"
    [ -f "$OUT/compiled/$s/$s.xmodel" ] || die "compiling stage $s did not produce $OUT/compiled/$s/$s.xmodel"

    # --! a stage must be exactly one DPU subgraph; anything else means an operator falls back to the CPU
    ndpu="$(grep -o 'DPU subgraph number [0-9]*' "$LOG" | tail -1 | grep -o '[0-9]*$' || true)"
    if [ -z "$ndpu" ]; then
        echo "[warn]: could not read the number of DPU subgraphs of stage $s from the compile log"
    elif [ "$ndpu" != 1 ]; then
        die "stage $s compiled to $ndpu DPU subgraphs, but a stage must be exactly one (see $LOG)"
    else
        echo "[build] stage $s: one DPU subgraph"
    fi
done

# ---------------------------------------------------------------------------------------
begin constants "export the constants (buffers and config values) for the board"
run python3 "$HOST/export_constants.py" "$OUT/model_plan.json" "$WEIGHTS" --out-consts "$OUT/model_consts.json"

# ---------------------------------------------------------------------------------------
begin golden "record the golden vectors of every stage (fake-quantized torch outputs)"
for s in $STAGES; do
    run python3 "$HOST/dump_golden.py" "$MODEL" "$CONFIG" "$s" "${COMMON[@]}" --out-golden "$OUT/golden/golden_$s.npz"
done

# ---------------------------------------------------------------------------------------
begin pipeline "verify the numpy glue against the torch model and record the pipeline golden"
run python3 "$HOST/check_pipeline.py" "$MODEL" "$CONFIG" "${COMMON[@]}" \
    --plan "$OUT/model_plan.json" --consts "$OUT/model_consts.json" --pipeline "$PIPELINE" \
    --labels "$LABELS" --golden "$OUT/golden/golden_pipeline.npz"

# ---------------------------------------------------------------------------------------
begin consistency "the goldens of the pipeline and of the single stages must agree"
GOLDENS=()
for s in $STAGES; do GOLDENS+=("$OUT/golden/golden_$s.npz"); done
run python3 "$HOST/check_goldens.py" "$OUT/golden/golden_pipeline.npz" "${GOLDENS[@]}"

# ---------------------------------------------------------------------------------------
begin bundle "assemble the board bundle"
B="$OUT/board"
mkdir -p "$B"

cp "$ROOT"/src/common/*.py "$ROOT"/src/board/*.py "$B/"
cp "$PIPELINE" "$B/pipeline.py"
cp "$OUT/model_plan.json" "$OUT/model_consts.json" "$B/"
for s in $STAGES; do
    cp "$OUT/compiled/$s/$s.xmodel" "$B/"
    cp "$OUT/golden/golden_$s.npz" "$B/"
done
cp "$OUT/golden/golden_pipeline.npz" "$B/"

DATA_FILE="$(basename "$DATA")"
LABELS_FILE="$(basename "$LABELS")"
cp "$DATA" "$B/$DATA_FILE"
cp "$LABELS" "$B/$LABELS_FILE"
cp "$ROOT/scripts/check_board.sh" "$B/"

cat > "$B/bridge_board.conf" << EOF
STAGES="$STAGES"
DATA_FILE="$DATA_FILE"
LABELS_FILE="$LABELS_FILE"
TRAJ_NSAMPLE=$TRAJ_NSAMPLE
EOF

# --! where did this bundle come from? The code lives on three machines and is not always moved with git, so
# --! the bundle carries a fingerprint of the sources it was built from: equal fingerprints, equal code.
tree_hashes() { ( cd "$1" && find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' | sort | xargs -r sha256sum ); }
SRC_ID="$( { echo src; tree_hashes "$ROOT/src"; echo scripts; tree_hashes "$ROOT/scripts"; echo example; tree_hashes "$EX"; } | sha256sum | cut -c1-12 )"

# --! git is reported only if BRIDGE is a repository of its own: inside another one (the Vitis-AI checkout,
# --! for example) 'git rev-parse' would name the revision of that repository
GIT_TOP="$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
if [ "$GIT_TOP" = "$ROOT" ]; then
    GIT_REV="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then GIT_DESC="$GIT_REV (uncommitted changes)"; else GIT_DESC="$GIT_REV (clean)"; fi
else
    GIT_DESC="not a git repository of its own"
fi
cat > "$B/BUILD_INFO.txt" << EOF
example      : $NAME
built        : $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)
source       : $SRC_ID (fingerprint of src/, scripts/ and the example folder)
git          : $GIT_DESC
stages       : $STAGES
architecture : $ARCH
python       : $(python3 --version 2>&1)
torch        : $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)
EOF

# --! checksums of every file of the bundle, so the board can tell whether it holds exactly this build
( cd "$B" && for f in *; do [ "$f" = MANIFEST.sha256 ] || echo "$f"; done | sort | xargs sha256sum > MANIFEST.sha256 )

echo "[build] bundle: $B ($(ls "$B" | wc -l | tr -d ' ') files, $(du -sh "$B" | cut -f1))"
echo
echo "[build] done. Next:"
echo "[build]   scripts/deploy_board.sh $EXAMPLE <user@board:/remote/dir> [--run]"
