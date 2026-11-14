#!/bin/sh
#
# check_board.sh -- run ON THE BOARD from the folder that scripts/deploy_board.sh filled.
#
#   sh check_board.sh
#
# 1. verifies that the files are exactly those of the build (MANIFEST.sha256)
# 2. checks every DPU stage against its golden vectors (check_golden.py)
# 3. runs the whole pipeline on the data and compares it with the host (run_pipeline.py)
#
# It stops with an error if any step fails.

cd "$(dirname "$0")" || exit 1
. ./bridge_board.conf

echo "==== bundle"
cat BUILD_INFO.txt
if command -v sha256sum > /dev/null 2>&1; then
    if sha256sum -c MANIFEST.sha256 > /dev/null 2>&1; then
        echo "manifest   : all files match the build"
    else
        echo "[error]: files in this folder differ from the build; show which with 'sha256sum -c MANIFEST.sha256'" >&2
        echo "[error]: copy the bundle again with scripts/deploy_board.sh" >&2
        exit 1
    fi
else
    echo "[warn]: sha256sum not available; the manifest was not verified"
fi

FAILED=0

for stage in $STAGES; do
    echo
    echo "==== stage $stage: DPU against the golden vectors"
    python3 check_golden.py "$stage.xmodel" "golden_$stage.npz" || FAILED=1
done

echo
echo "==== pipeline: DPU stages plus numpy glue, on all trajectories"
python3 run_pipeline.py pipeline.py model_plan.json model_consts.json \
    --xmodels . --data "$DATA_FILE" --traj-nsample "$TRAJ_NSAMPLE" \
    --labels "$LABELS_FILE" --golden golden_pipeline.npz || FAILED=1

echo
if [ "$FAILED" = 0 ]; then
    echo "[board] all steps completed"
else
    echo "[board] SOME STEPS FAILED" >&2
    exit 1
fi
