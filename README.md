# BEVDepth 
BEVDepth learns depth estimation from LiDAR-supervised data, but performs 3D object detection from camera images alone. This repo is the fork of [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) with a conflict-free Docker stack.
(Torch 1.12/cu116, mmcv-full 1.6,mmdet 2.25, mmdet3d 1.0.0rc4, mmseg 0.26)

---

## Additions 

| Item                                                                      | Purpose                                                                        |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `docker/Dockerfile`                                                       | For image build.                          |
| `requirements.txt` & `constraints.txt`                                    | All Python package versions pinned to avoid conflicts.                         |
| `scripts/gen_info.py`                                                     | Modified: accepts `--dataroot` and `--save_dir` for output locations. |
| `bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py` | Modified experiment file for correct dataset paths and unique experiment name. |
---

## Quick Start Guide

### 1️⃣ Build Docker Image

```bash
cd docker
docker build -t bevdepth:original .
```

### 2️⃣ Create and Run Docker Container

```bash
docker run --gpus all -it \
  --name bevdepth-original \
  --shm-size 16g \
  -v "$HOME/BEVDepth:/workspace/BEVDepth" \
  -v "/media/prabuddhi/Crucial X92/bevfusion-main/data/nuscenes:/workspace/data/nuScenes" \
  bevdepth:original /bin/bash
```

- Update the GitHub folder path and dataset path as per your local directory.

### 3️⃣ Restart Existing Container

```bash
docker start -ai bevdepth:original
```

---

## One-Time Setup Inside Docker

### 0️⃣ Optional: Verify CUDA & Torch

```bash
cd /workspace/BEVDepth
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

python - <<'PY'
import torch; print(torch.__version__, torch.version.cuda)
PY
# Expected: 1.12.1  11.6
```

### 1️⃣ Generate NuScenes Metadata

```bash
mkdir -p /workspace/data/nuScenes_BEVDepth
python scripts/gen_info.py \
  --dataroot /workspace/data/nuScenes \
  --save_dir /workspace/data/nuScenes_BEVDepth
```

### 2️⃣ Download Pretrained Weights

```bash
mkdir -p pretrained
wget https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth \
  -P pretrained/
```

### 3️⃣ (Optional) Re-compile CUDA Ops (only if upgrading Torch/CUDA)

```bash
# python setup.py clean --all && python setup.py develop --no-deps
```

---

## 🚦 Running BEVDepth

### Sanity Check: Evaluation

```bash
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  --ckpt_path pretrained/bev_depth_lss_r50_256x704_128x128_24e_2key.pth \
  --gpus 1 -b 1 -e
```

> Expected: mAP ≈ 0.33 / NDS ≈ 0.44

### Training / Fine-tuning

```bash
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  --amp_backend native \
  --gpus 1 -b 1
```

- Increase `-b` and `--gpus` based on hardware availability.

---

## 📝 Code Changes Summary

| File                                            | Change                                             | Reason                           |
| ----------------------------------------------- | -------------------------------------------------- | -------------------------------- |
| `scripts/gen_info.py`                           | Added `argparse` for `--dataroot` and `--save_dir` | Flexible output directories      |
| `bev_depth_lss_r50_256x704_128x128_24e_2key.py` | Adjusted experiment name, dataset path             | Clean experiment structure       |
| All other code                                  | Unchanged                                          | Only dataset path logic modified |

---

## 🤩 Library Versions (Fully Frozen)

```
torch-1.12.1+cu116        mmcv-full-1.6.0
torchvision-0.13.1+cu116  mmdet-2.25.0
torchaudio-0.12.1+cu116   mmdet3d-1.0.0rc4
cuda-11.6 runtime         mmsegmentation-0.26.0
numba-0.56.4              llvmlite-0.39.1
```

> Rebuild = exact reproduction.

---

## 🔧 Troubleshooting

| Issue                             | Solution                                                           |
| --------------------------------- | ------------------------------------------------------------------ |
| MMCV version conflict             | `pip install mmcv-full==1.6.0 -f ...`                              |
| CUDA handle errors                | Ensure Torch/CUDA versions match image.                            |
| Missing voxel\_pooling\_inference | `python setup.py clean --all && python setup.py develop --no-deps` |
| VS Code remote can't see packages | Always attach to the container correctly (`bevdepth-cu116`).       |

---

## 📄 Citation

```bibtex
@article{li2022bevdepth,
  title={BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection},
  author={Li, Yinhao and Ge, Zheng and ...},
  journal={arXiv preprint arXiv:2206.10092},
  year={2022}
}
```

---

*Happy BEV research! 🚀 — Feel free to open issues or PRs for updates or fixes.*
