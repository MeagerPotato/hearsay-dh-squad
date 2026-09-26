#!/bin/bash
# Kill a job (by pgrep pattern) if system-wide free memory drops below a threshold. Protects the laptop.
PATTERN="$1"; MIN_FREE="${2:-12}"
while pgrep -f "$PATTERN" >/dev/null; do
  free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ]; then
    echo "$(date +%T) free memory ${free}% < ${MIN_FREE}% -> killing $PATTERN"; pkill -f "$PATTERN"; exit 1
  fi
  sleep 10
done
echo "$(date +%T) $PATTERN finished; watchdog exiting"
