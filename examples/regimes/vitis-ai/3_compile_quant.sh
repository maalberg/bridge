#!/bin/bash

# --! there must be model class parameter
if [ $# -lt 1 ]; then
    echo "[usage]: $0 <model_classname>"
    exit 1
fi

vai_c_xir -x "quantize_result/$1_int.xmodel" -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -o compiled -n "$1"
