# Installation

Python 3.11 is recommended. Create a Conda environment, then install the pinned
training dependencies:

```bash
conda create -n vtla python=3.11 -y
conda activate vtla
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
export PYTHONPATH=$PWD
```

The dependency set targets CUDA-enabled training. Select PyTorch, JAX, and CUDA versions compatible with the local accelerator and driver.

Apply the bundled Transformers replacements required by the PyTorch model:

```bash
cp -r n0vtla/models_pytorch/transformers_replace/* \
  "$CONDA_PREFIX/lib/python3.11/site-packages/transformers/"
```

The tactile encoder uses DINOv2. Cache it once before using the launcher's default offline mode.

`train.sh` runs offline and looks for the cache under `$N0VTLA_DATA_HOME/huggingface`, where
`N0VTLA_DATA_HOME` defaults to `<repo>/models`. Export the same `HF_HOME` when caching, or the
launcher's preflight will fail to find the model even though it is on disk:

```bash
export N0VTLA_DATA_HOME="$PWD/models"
export HF_HOME="$N0VTLA_DATA_HOME/huggingface"

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PY'
from transformers import AutoModel

AutoModel.from_pretrained("facebook/dinov2-base")
PY
```

If you would rather use the standard `~/.cache/huggingface`, export `HF_HOME` to it before
running `train.sh` as well; the launcher honours an `HF_HOME` you set yourself.
