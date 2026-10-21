# BRIDGE

**B**oard-**R**eady **I**nference, **D**eployment, **G**uidance and **E**xplanation

For scientists and ML researchers who need FPGA deployment but don't want to become FPGA engineers.

> **Status:** early work in progress. Currently targets the AMD/Xilinx Vitis AI flow (DPU) on the ZCU104 board.

## Repository layout

```
BRIDGE/
├── vitis-ai/                  # Shared, example-independent scripts
│   ├── quant_calib.py         #   quantization calibration
│   └── quant_test.py          #   quantized-model accuracy test
└── examples/
    └── regimes/               # One folder per example
        ├── *.py               #   model + dataset code (see "Example contract")
        ├── *.pth              #   trained weights and biases
        ├── *.csv              #   example data
        └── vitis-ai/          #   thin shell wrappers around ../../../vitis-ai/*.py
            ├── quant_calib.sh
            └── quant_test.sh
```

The idea: the generic logic lives once, in the top-level `vitis-ai/` folder. Each example only provides its own model, dataset, weights and data, plus tiny shell wrappers that point the generic scripts at that example.

## Running an example

From inside an example's `vitis-ai/` folder (shown here for `regimes`), inside a docker container with Vitis AI PyTorch:

```bash
cd examples/regimes/vitis-ai
./1_quant_calib.sh <model_classname>   # calibrate and export the quantized model
./2_quant_test.sh  <model_classname>   # compare float vs. quantized accuracy
./3_compile_quant.sh <model_classname> # compile the quantized model
```

`<model_classname>` is the name of the model class defined in the example's Python code.

## Adding a new example

1. Create `examples/<your_example>/`.
2. Add your Python code with a model class and a dataset class, the trained `.pth` weights, and your data.
3. Copy the `vitis-ai/` wrapper folder from an existing example (e.g. `regimes`). The wrappers locate the shared scripts relative to their own location, so no path edits should be needed.
4. Run the wrappers as shown above.

### Example contract

For the shared scripts to work, an example must provide:

- a `torch.nn.Module` model class, loadable by name
- a dataset class that yields `(input, label)` pairs
- trained weights (`.pth`) matching the model class
- data files (`.csv`) readable by the dataset class

*(This contract will be formalized as abstract interfaces once a second example exists.)*
