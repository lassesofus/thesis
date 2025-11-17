#!/bin/bash

# Directory where VS Code Remote SSH stores workspace data
ROOT="$HOME/.vscode-server/data/User/workspaceStorage"

# Threshold in MB above which a workspace folder is considered broken
THRESHOLD_MB=200

echo "Scanning $ROOT for oversized workspace folders..."

for dir in "$ROOT"/*; do
    [ -d "$dir" ] || continue

    size_mb=$(du -sm "$dir" | cut -f1)

    if [ "$size_mb" -gt "$THRESHOLD_MB" ]; then
        echo "Deleting $dir ($size_mb MB)"
        rm -rf "$dir"
    fi
done

echo "Done."
