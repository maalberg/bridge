# BRIDGE

**B**oard-**R**eady **I**nference, **D**eployment, **G**uidance and **E**xplanation

For scientists and ML researchers who need FPGA deployment but don't want to become FPGA engineers.

> **Status:** early work in progress. Currently targets the AMD/Xilinx Vitis AI flow (DPU) on the ZCU104 board.

## Repository layout

```
BRIDGE/
├── scripts/
│   ├── build_host.sh      regenerate every artifact of an example (container)
│   ├── deploy_board.sh    copy the board bundle to the board (any machine with ssh)
│   └── check_board.sh     verify and run the bundle (on the board; ends up inside the bundle)
├── src/
│   ├── common/            numpy + stdlib only, used on host AND board
│   ├── board/             numpy only (vart, xir on the board)
│   └── host/              torch + vai_q_pytorch, runs in the container
├── examples/
│   └── regimes/           one folder per example
│       ├── bridge.conf    its parameters: model file, weights, data, labels, samples per trajectory
│       ├── model.py, model_config.json, pipeline.py
│       ├── data/          trajectories.csv, labels.csv
│       └── weights/       model.pth
└── build/<example>/       generated, not in git (see below)
```

Two rules keep the board side small and honest:

* nothing in `src/common` or `src/board` may import torch; the board bundle is made of exactly these two folders
  plus the example's `pipeline.py`;
* code that runs in the container must stay compatible with Python 3.7 (no walrus operator, no `ast.unparse`, ...).

## Setup for ZCU104 with PyTorch Vitis AI 3.0

> **Status:** The following commands were tested on Ubuntu 20.04.6 LTS.

1. Install docker

    ```
    sudo apt update
    sudo apt install -y docker.io
    sudo usermod -aG docker $USER
    ```

2. Clone version 3.0 of Vitis-AI repository

    ```
    git clone --branch v3.0 https://github.com/Xilinx/Vitis-AI
    ```

3. Pull PyTorch CPU container

    ```
    docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
    ```

4. Clone BRIDGE repo into the previously cloned Vitis-AI folder

    ```
    git clone https://github.com/maalberg/bridge.git
    ```

5. Launch PyTorch CPU container

    ```
    ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
    ```

6. Inside the container: activate conda environment

    ```
    conda activate vitis-ai-pytorch
    ```

7. Inside the container: build an example (regimes)

    ```
    cd bridge
    ./scripts/build_host.sh examples/regimes
    ```

8. Inside the container: deploy the build on the board

    ```
    ./scripts/deploy_board.sh --run examples/regimes petalinux@<board-ip>:~/bridge/regimes
    ```

## Command details

In the container (`conda activate vitis-ai-pytorch`):

```
git pull
scripts/build_host.sh examples/regimes          # -v shows full output, --skip-test skips the slow quantized test
```

From inside the container, or anywhere that can reach the board via ssh:

```
scripts/deploy_board.sh --run examples/regimes petalinux@<board-ip>:~/bridge/regimes
```

The bundle travels as a tar archive through one ssh connection (one password prompt; only ssh and tar are needed
on either side). `BRIDGE_SSH_OPTS="-i ~/.ssh/board_key"` passes options on to ssh.

or, on the board, from the bundle folder:

```
sh check_board.sh
```

## What the build does

`build_host.sh` deletes `build/<example>/` and then runs, in this order, stopping at the first failure:

1. `analyze_model.py` : plan (stages, row shapes, constants)
2. `calib_quant.py` : once per stage, in its own process
3. `test_quant.py` : quantized composite end to end, with accuracy and label alignment (informational)
4. `export_quant.py` : once per stage, in its own process; the expected `<Class>_int.xmodel` must exist
5. `vai_c_xir` : once per stage; every stage must compile to exactly one DPU subgraph
6. `export_constants.py` : buffers and config values
7. `dump_golden.py` : golden vectors per stage
8. `check_pipeline.py` : numpy glue against the torch model, plus the pipeline golden; any failing check stops the build
9. `check_goldens.py` : the goldens of the pipeline and of the single stages must agree (a stage must be quantized the
   same way alone and in the pipeline)
10. bundle: `build/<example>/board/`

Every step writes `build/<example>/logs/NN_<step>.log`.

## The bundle

`build/<example>/board/` is flat and complete: the board code, `pipeline.py`, the plan, the constants, one
`<stage>.xmodel` and `golden_<stage>.npz` per stage, `golden_pipeline.npz`, the data and labels, `check_board.sh`,
and two files about the bundle itself:

* `BUILD_INFO.txt`: when and where it was built, a fingerprint of the sources (equal fingerprints mean equal code,
  with or without git), the git revision if BRIDGE is a repository of its own, and the tool versions;
* `MANIFEST.sha256`: checksums of every other file. `check_board.sh` verifies it first, so a board holding files
  from two different builds refuses to run.

## Quantization diagnostics: which tensor's range costs the accuracy?

`scripts/ablate_host.sh` (`src/host/ablate_quant.py`) takes the FLOAT model and quantizes one boundary tensor of the stages at a time
(8 bits, power-of-two scale, saturation, as the DPU does), everything else staying float:

```
scripts/ablate_host.sh examples/regimes \
    --fix encoder.in=6,encoder.out=6,classifier.in=7,classifier.out=5 --reference 94.31
```

`--fix` gives the fix points of the deployed model (`check_golden.py` prints them on the board), `--reference` the accuracy measured on
the board. It reports (1) what every tensor carries and which fix point would avoid saturation, (2) the accuracy for a sweep of fix
points per tensor, and (3) with `--fix`: all tensors at the deployed fix points, one tensor kept in float, one tensor with a range
that avoids saturation. The share of values that saturate does not say whether it hurts (a threshold may lie well inside the range):
the accuracy does. Only the boundary tensors are emulated, not the weights or the activations inside a stage; the last lines say how
much of the board's loss the boundary tensors explain. Results go to `build/<example>/ablation.json`.

## When the board and the host disagree

`check_board.sh` ends by comparing trajectory 0 with the host, step by step, in pipeline order:

```
raw trajectory                      the data file read on the board
raw windows (make_inputs)           the windows cut from it
stage <name>: input rows            what the board feeds the stage (normalization included)
stage <name>: output, host rows in  the stage on the HOST's input rows: does the DPU reproduce the host here?
```

The first line that is not 100 % identical shows where to look: the data file or numpy (first two lines), the
constants or the glue (input rows), or the DPU stage itself (output on the host's rows).

## Adding an example

Copy `examples/regimes`, adapt `bridge.conf`, write `model.py` (class `model` with `make_inputs` and `align_labels`),
`model_config.json` and `pipeline.py`, put the weights in `weights/model.pth`, run `scripts/build_host.sh`.
The stages are not listed anywhere: they are the child modules of `model` that the plan finds running.

## Settings

* `BRIDGE_ARCH` : the DPU architecture file for `vai_c_xir`
  (default `/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json`).

