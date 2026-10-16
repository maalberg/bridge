#!/bin/bash

# --! there must be model class parameter
if [ $# -lt 1 ]; then
    echo "[usage]: $0 <model_classname>"
    exit 1
fi

# --! absolute path of the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

# --! path to example directory relative to script directory
EXAMPLE_DIR="$SCRIPT_DIR/../"

# --! navigate to python script
PYTHON_SCRIPT="$SCRIPT_DIR/../../../vitis-ai/quant_calib.py"

# --! invoke python script
python3 "$PYTHON_SCRIPT" "$EXAMPLE_DIR" "$@"
