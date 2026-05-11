#!/usr/bin/env bash
# auto_save.sh — DGX-side rsync of low-confidence active-learning frames.
#
# Pulls JPG + sidecar JSON files from a remote (Windows production) hot folder
# into the canonical `~/vehicle_dataset/raw/active_learning/<YYYY-MM-DD>/`
# location so they can re-enter Phase 2 labelling in Label Studio.
#
# The remote hot folder is set in scene_config.json `active_learning.hot_dir`
# on the production box (default: `~/al_hot`). On the DGX, the destination
# defaults to `~/vehicle_dataset/raw/active_learning/`.
#
# Usage
# -----
#     auto_save.sh <user@host:path>            # explicit remote
#     AL_REMOTE=user@host:path auto_save.sh    # via env
#
# Optional env:
#     AL_DEST=/path/to/dest_root   (default: ~/vehicle_dataset/raw/active_learning)
#     AL_SSH_KEY=/path/to/key      (passed to rsync -e "ssh -i ...")
#
# Cron-friendly: idempotent, removes source files after a successful copy so
# the hot folder doesn't grow unbounded. Logs to /tmp/auto_save.log.
#
# Suggested crontab line (DGX): every 15 min
#     */15 * * * * AL_REMOTE=admin@winbox:~/al_hot/ \
#         /bin/bash /home/admin/vehicle-counting-app/tools/auto_save.sh \
#         >> /tmp/auto_save.log 2>&1

set -uo pipefail

REMOTE="${1:-${AL_REMOTE:-}}"
DEST_ROOT="${AL_DEST:-$HOME/vehicle_dataset/raw/active_learning}"
DATE="$(date +%F)"
DEST="$DEST_ROOT/$DATE"
LOG="/tmp/auto_save.log"
SSH_OPT=""

if [ -z "$REMOTE" ]; then
  echo "[auto_save] usage: $0 <user@host:path>  (or set AL_REMOTE)" >&2
  exit 64
fi

if [ -n "${AL_SSH_KEY:-}" ]; then
  SSH_OPT='-e "ssh -i '"$AL_SSH_KEY"' -o BatchMode=yes -o StrictHostKeyChecking=accept-new"'
else
  SSH_OPT='-e "ssh -o BatchMode=yes"'
fi

mkdir -p "$DEST"

echo "[auto_save] $(date) src=$REMOTE  dst=$DEST" >> "$LOG"

# rsync only .jpg + .json from the remote's <YYYY-MM-DD>/ subdir, removing
# source files after a successful copy. --include order matters for --exclude.
eval rsync -av --remove-source-files \
  --include='*/' \
  --include='*.jpg' \
  --include='*.json' \
  --exclude='*' \
  $SSH_OPT \
  "${REMOTE%/}/$DATE/" "$DEST/" >> "$LOG" 2>&1

rc=$?
if [ "$rc" -eq 0 ]; then
  pulled=$(find "$DEST" -newer "$DEST" -name '*.jpg' 2>/dev/null | wc -l)
  echo "[auto_save] OK  rc=$rc  files-in-dest=$(find "$DEST" -name '*.jpg' | wc -l)" >> "$LOG"
elif [ "$rc" -eq 23 ] || [ "$rc" -eq 24 ]; then
  # 23 = partial transfer (some files vanished), 24 = source files vanished —
  # both can happen if the remote is mid-write, treat as warning not failure.
  echo "[auto_save] WARN partial transfer rc=$rc" >> "$LOG"
else
  echo "[auto_save] FAIL rc=$rc" >> "$LOG"
  exit "$rc"
fi
