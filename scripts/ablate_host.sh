#!/usr/bin/env bash
#
# ablate_host.sh -- which tensor's fixed-point range costs the accuracy? (see src/host/ablate_quant.py)
#
#   scripts/ablate_host.sh <example_dir> [options of ablate_quant.py]
#
# Run it after scripts/build_host.sh, in the same environment. It takes the model, weights, data and labels from
# <example_dir>/bridge.conf and the plan and constants from build/<example>/, and passes every further option on,
# usually the fix points of the deployed model (printed by check_golden.py on the board) and the accuracy that the
# board measured:
#
#   scripts/ablate_host.sh examples/regimes \
#       --fix encoder.in=6,encoder.out=6,classifier.in=7,classifier.out=5 --reference 94.31
#
# Results: build/<example>/ablation.json and build/<example>/logs/ablation.log.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() { echo "[ablate] error: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: $0 <example_dir> [options of ablate_quant.py]"
EXAMPLE="$1"
shift
[ -d "$EXAMPLE" ] || die "example directory $EXAMPLE does not exist"
EX="$(cd "$EXAMPLE" && pwd -P)"
[ -f "$EX/bridge.conf" ] || die "$EX/bridge.conf does not exist"

# shellcheck disable=SC1091
. "$EX/bridge.conf"
for v in MODEL CONFIG PIPELINE WEIGHTS DATA LABELS TRAJ_NSAMPLE; do
    [ -n "${!v:-}" ] || die "$v is not set in $EX/bridge.conf"
done

NAME="$(basename "$EX")"
OUT="$ROOT/build/$NAME"
[ -f "$OUT/model_plan.json" ] && [ -f "$OUT/model_consts.json" ] \
    || die "$OUT has no plan or constants; run scripts/build_host.sh $EXAMPLE first"

export PYTHONPATH="$ROOT/src/common:$ROOT/src/board:$ROOT/src/host${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/logs"

python3 "$ROOT/src/host/ablate_quant.py" "$EX/$MODEL" "$EX/$CONFIG" \
    --weights "$EX/$WEIGHTS" --data "$EX/$DATA" --traj-nsample "$TRAJ_NSAMPLE" --labels "$EX/$LABELS" \
    --plan "$OUT/model_plan.json" --consts "$OUT/model_consts.json" --pipeline "$EX/$PIPELINE" \
    --out "$OUT/ablation.json" "$@" 2>&1 | tee "$OUT/logs/ablation.log"
