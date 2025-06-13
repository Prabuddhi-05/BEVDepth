# BEVDepth 
BEVDepth learns depth estimation from LiDAR-supervised data, but performs 3D object detection from camera images alone. This repo is the fork of [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) with a conflict-free Docker stack.
(Torch 1.12/cu116, mmcv-full 1.6,mmdet 2.25, mmdet3d 1.0.0rc4, mmseg 0.26)

---

## Additions 

| Item                                                                      | Purpose                                                                        |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `docker/Dockerfile`                                                       | For image build (OS + CUDA + Python layer)                          |
| `requirements.txt`                                                        | Python package runtime dependencies.                         |
| `constraints.txt`                                                         | PyTorch and related packages to match CUDA version                         |
| `scripts/gen_info.py`                                                     | Modified: accepts `--dataroot` and `--save_dir` for output locations. |
| `bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py` | Modified the paths for generated infos. |
---

    This is a widely recommended practice for any research-grade repo.

File	Purpose
Dockerfile	OS + CUDA + Python layer
requirements.txt	Python-level runtime dependencies
constraints.txt	Exact version pinning to guarantee reproducibility

## Quick start guide

### Build Docker image

```bash
cd docker
docker build -t bevdepth:original .
```

### Create and run Docker container

```bash
docker run --gpus all -it \
  --name bevdepth-original \
  --shm-size 16g \
  -v "$HOME/BEVDepth:/workspace/BEVDepth" \
  -v "/media/prabuddhi/Crucial X92/bevfusion-main/data/nuscenes:/workspace/data/nuScenes" \
  bevdepth:original /bin/bash
```

- Update the GitHub project folder path and dataset path to match your local directory structure.

### Restart existing container

```bash
docker start -ai bevdepth:original
```

---

## One-time setup inside Docker

### Set CUDA environment variables

```bash
cd /workspace/BEVDepth
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
```
### Verify installation (Optional) 
```bash
python - <<'PY'
import torch; print(torch.__version__, torch.version.cuda)
PY
# Expected: 1.12.1  11.6

python - <<'PY'
import torch, mmcv, mmdet, mmdet3d, mmseg
print("torch      :", torch.__version__, "CUDA", torch.version.cuda)
print("mmcv-full  :", mmcv.__version__)
print("mmdet      :", mmdet.__version__)
print("mmdet3d    :", mmdet3d.__version__)
print("mmseg      :", mmseg.__version__)
PY
# Expected:
# torch      : 1.12.1+cu116 CUDA 11.6
# mmcv-full  : 1.6.0
# mmdet      : 2.25.0
# mmdet3d    : 1.0.0rc4
# mmseg      : 0.26.0

python - <<'PY'
import numba, llvmlite, numpy as np
print("numba    :", numba.__version__)
print("llvmlite :", llvmlite.__version__)
print("numpy    :", np.__version__)
PY
# Expected:
# numba    : 0.56.4
# llvmlite : 0.39.1

python - <<'PY'
from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference
print("Wrapper OK :", voxel_pooling_inference)          # should be a <function …>
PY
```

### Generate NuScenes metadata

```bash
mkdir -p data
ln -s /workspace/data/nuScenes data/nuScenes     
mkdir -p /workspace/data/nuScenes_BEVDepth
ln -s /workspace/data/nuScenes_BEVDepth ./data/nuScenes_BEVDepth
cd /workspace/BEVDepth
python scripts/gen_info.py \
  --dataroot /workspace/data/nuScenes \
  --save_dir /workspace/data/nuScenes_BEVDepth
```

### Download pretrained weights

```bash
mkdir -p pretrained
wget https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth \
  -P pretrained/
```
### Compile

```bash
python setup.py develop 
```

### Re-compile CUDA Ops (only if upgrading Torch/CUDA) (Optional)

```bash
python setup.py clean --all
rm -rf build/ BEVDepth.egg-info/
python setup.py develop --no-deps
```
---

## Running BEVDepth

### Evaluation

```bash
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  --ckpt_path pretrained/bev_depth_lss_r50_256x704_128x128_24e_2key.pth \
  --gpus 1 -b 1 -e
```

> Expected: mAP ≈ 0.33 / NDS ≈ 0.44

### Training

```bash
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  --amp_backend native \
  --gpus 1 -b 1
```

- Increase `-b`(batch size - 1 sample per gpu) and `--gpus` based on hardware availability.

---

## Library versions

```
torch-1.12.1+cu116        mmcv-full-1.6.0
torchvision-0.13.1+cu116  mmdet-2.25.0
torchaudio-0.12.1+cu116   mmdet3d-1.0.0rc4
cuda-11.6 runtime         mmsegmentation-0.26.0
numba-0.56.4              llvmlite-0.39.1
```
