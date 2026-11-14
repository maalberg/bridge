#!/usr/bin/env bash
#
# deploy_board.sh -- copy the board bundle of an example to the board.
#
#   scripts/deploy_board.sh [--run] <example_dir> <user@board:/remote/dir>
#
#   --run   afterwards run check_board.sh on the board, in the same ssh connection
#
# The bundle (build/<example>/board/) is streamed as a tar archive through ONE ssh connection, so
# nothing but ssh and tar is needed on either side (no rsync, no scp: recent scp refuses the
# 'directory/.' source that an earlier version of this script used) and there is one password
# prompt. A leading ~ in the remote directory is expanded on the board. Files already in the remote
# directory are overwritten; others are left alone, so use a directory of its own.
#
# Options for ssh (a key, for example) can be given in the environment variable BRIDGE_SSH_OPTS:
#   BRIDGE_SSH_OPTS="-i ~/.ssh/board_key" scripts/deploy_board.sh ...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() { echo "[deploy] error: $*" >&2; exit 1; }

RUN=0
POS=()
for arg in "$@"; do
    case "$arg" in
        --run)     RUN=1 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        -*)        die "unknown option $arg" ;;
        *)         POS+=("$arg") ;;
    esac
done
[ "${#POS[@]}" -eq 2 ] || die "usage: $0 [--run] <example_dir> <user@board:/remote/dir>"

EXAMPLE="${POS[0]}"
TARGET="${POS[1]}"
[ -d "$EXAMPLE" ] || die "example directory $EXAMPLE does not exist"
NAME="$(basename "$(cd "$EXAMPLE" && pwd -P)")"
BUNDLE="$ROOT/build/$NAME/board"
[ -f "$BUNDLE/MANIFEST.sha256" ] || die "no bundle at $BUNDLE; run scripts/build_host.sh $EXAMPLE first"

case "$TARGET" in
    *:*) ;;
    *)   die "the target must look like user@board:/remote/dir" ;;
esac
REMOTE_HOST="${TARGET%%:*}"
REMOTE_DIR="${TARGET#*:}"
[ -n "$REMOTE_DIR" ] || die "the target has no remote directory"

# --! a ~ must be expanded by the shell on the board, not quoted away: pass it on as $HOME
case "$REMOTE_DIR" in
    "~")   REMOTE_DIR='$HOME' ;;
    "~/"*) REMOTE_DIR='$HOME/'"${REMOTE_DIR#\~/}" ;;
esac

read -r -a SSH_OPTS <<< "${BRIDGE_SSH_OPTS:-}"

REMOTE_CMD="mkdir -p \"$REMOTE_DIR\" && tar -xf - -C \"$REMOTE_DIR\""
if [ "$RUN" = 1 ]; then
    REMOTE_CMD="$REMOTE_CMD && cd \"$REMOTE_DIR\" && sh check_board.sh"
fi

echo "[deploy] $BUNDLE -> $REMOTE_HOST:$REMOTE_DIR"
echo "[deploy] $(grep -E '^(built|source)' "$BUNDLE/BUILD_INFO.txt" | tr -s ' ' | tr '\n' ';')"

tar -C "$BUNDLE" -cf - . | ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$REMOTE_HOST" "$REMOTE_CMD"

if [ "$RUN" = 1 ]; then
    echo "[deploy] done"
else
    echo "[deploy] done; on the board:  cd $REMOTE_DIR && sh check_board.sh"
fi
