## DeepClean Environment: `dc-prod-v4-cu128`

This document describes how to install the DeepClean environment based on `micromamba`, Python 3.12, PyTorch 2.7.1, and CUDA 12.8 wheels.

This environment has been tested on:

- NVIDIA A100 / `sm_80`
- NVIDIA RTX PRO 4000 Blackwell / `sm_120`

---

## 1. Install micromamba

If `micromamba` is not installed, install it with:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

If `curl` fails with a library/OpenSSL symbol error, the shell may have a polluted `LD_LIBRARY_PATH`. In that case, download the installer with a clean environment:

```bash
env -i HOME="$HOME" PATH="/usr/local/bin:/usr/bin:/bin" \
  /usr/bin/curl -L https://micro.mamba.pm/install.sh -o /tmp/micromamba_install.sh

env -i HOME="$HOME" PATH="/usr/local/bin:/usr/bin:/bin" SHELL="$SHELL" \
  bash /tmp/micromamba_install.sh
```

After installation, restart the shell or run:

```bash
source ~/.bashrc
```

---

## 2. Create the environment

Use:

```bash
/home/shuwei.yeh/deepclean-v1/deepclean-prod-O4a/environment-blackwell.yml
```

Create and activate the environment:

```bash
cd /home/shuwei.yeh/deepclean-v1/deepclean-prod-O4a

micromamba env create -f environment-blackwell.yml
micromamba activate dc-prod-v4-cu128
```

The environment file should include the GWF backend packages:

```yaml
  - lalframe
  - python-lalframe
```

These are required by `gwpy` to read `.gwf` files.

---

## 3. Install PyTorch CUDA 12.8 wheels

After activating the environment:

```bash
python -m pip install --upgrade pip

python -m pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install "torchsummary>=1.5,<2"
```

Then install DeepClean in editable mode:

```bash
cd /home/shuwei.yeh/deepclean-v1/deepclean-prod-O4a
python -m pip install --no-build-isolation -e .
```

Use `--no-build-isolation` because the current DeepClean packaging may otherwise fail due to missing `pkg_resources` inside pip’s isolated build environment.

---

## 4. Verify PyTorch and CUDA

Activate the environment and limit BLAS/OpenMP threads:

```bash
micromamba activate dc-prod-v4-cu128

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MALLOC_ARENA_MAX=2
```

Run the CUDA test:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("arch list", torch.cuda.get_arch_list())

x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("matmul ok", y.mean().item())

conv = torch.nn.Conv1d(6, 8, 7, padding=3).cuda()
inp = torch.randn(8, 6, 4096, device="cuda")
out = conv(inp)
torch.cuda.synchronize()
print("conv1d ok", out.shape)
PY
```

Expected example on A100:

```text
torch 2.7.1+cu128
cuda runtime 12.8
cuda available True
device NVIDIA A100-SXM4-80GB
capability (8, 0)
```

Expected example on Blackwell:

```text
torch 2.7.1+cu128
cuda runtime 12.8
cuda available True
device NVIDIA RTX PRO 4000 Blackwell SFF Edition
capability (12, 0)
```

The architecture list should include:

```text
sm_80
sm_90
sm_120
compute_120
```

---

## 5. Verify GWF frame support

```bash
python -c "from gwpy.timeseries.io.gwf import get_default_gwf_api; print(get_default_gwf_api())"
```

A valid output is usually:

```text
lalframe
```

If this fails with:

```text
ImportError: no GWF API available
```

install the missing backend:

```bash
micromamba install -c conda-forge python-lalframe lalframe
```

Then update `environment-blackwell.yml` accordingly.

---

## 6. Verify DeepClean import

```bash
python -c "import deepclean_prod; import gwpy; import torch; print('DeepClean import OK')"
```

Expected output:

```text
DeepClean import OK
```