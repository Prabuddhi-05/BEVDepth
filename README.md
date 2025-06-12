# BEVDepth - CUDA 11.6 / Torch 1.12 Reproducible Environment  🚘🔭

> **Bird’s-Eye-View depth & 3D object detection (camera + LiDAR)**  
> Fork of [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)  
> with a fresh, conflict-free Docker stack (Torch 1.12/cu116, mmcv-full 1.6,  
> mmdet 2.25, mmdet3d 1.0.0rc4, mmseg 0.26) and fully scripted setup.
> 
> ---

## 📂 What’s inside

| Folder / file                    | Purpose |
|----------------------------------|---------|
| `docker/Dockerfile`              | **Single-stage image** – builds all deps & compiles CUDA ops. |
| `requirements.txt` + `constraints.txt` | Pin “generic” Python deps & force Torch 1.12/cu116 wheels. |
| `scripts/gen_info.py`            | Minor patch: now takes `--dataroot` & `--save_dir` args so you can write `.pkl` files anywhere. |
| `bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py` | **Tiny path edit** (see “Code changes” below) so the experiment reads data from the new location. |
| `README.md` (this file)          | All steps on one page. |

---

## 🐳 Quick-start (TL;DR)

```bash
# 1. build the image (≈ 8 min first time)
cd docker
docker build -t bevdepth:cu116 .

# 2. launch a named dev container
docker run --gpus all -it \
  --name bevdepth-cu116 \
  --shm-size 16g \
  -v "$HOME/BEVDepth:/workspace/BEVDepth" \
  -v "/media/prabuddhi/Crucial X92/bevfusion-main/data/nuscenes:/workspace/data/nuScenes" \
  bevdepth:cu116
```

> **Already created the container?**  
> ```
> docker start -ai bevdepth-cu116
> ```

---

## ⚙️ One-time setup *inside* the container

```bash
# 0. (optional) verify Torch/CUDA
python - <<'PY'
import torch; print(torch.__version__, torch.version.cuda)
PY
# → 1.12.1  11.6

# 1. generate NuScenes metadata
mkdir -p /workspace/data/nuScenes_BEVDepth
python scripts/gen_info.py \
        --dataroot /workspace/data/nuScenes \
        --save_dir /workspace/data/nuScenes_BEVDepth

# 2. download a pretrained checkpoint
mkdir -p pretrained
wget https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth \
     -P pretrained/

# 3. (re)compile CUDA ops if you ever upgrade Torch/CUDA
#    (NOT needed after first docker build)
# export CUDA_HOME=/usr/local/cuda
# python setup.py clean --all && python setup.py develop --no-deps
```

---

## 🚦 Run!

### Evaluation sanity check
```bash
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
       --ckpt_path pretrained/bev_depth_lss_r50_256x704_128x128_24e_2key.pth \
       --gpus 1 -b 1 -e
```

You should see metrics ending in **mAP ≈ 0.33 / NDS ≈ 0.44** – identical to the paper.

### Training from scratch / fine-tune
```bash
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
       --amp_backend native \
       --gpus 1 -b 1
```
*(increase `-b` and `--gpus` when you have more memory / GPUs).*

---

## 📝 Code changes

| File | Line(s) | Why |
|------|---------|-----|
| `scripts/gen_info.py` | new `argparse` block + replaces hard‑coded `./data/nuScenes` with `args.dataroot`, and writes pkl to `args.save_dir`. | write metadata anywhere without editing code each time. |
| `bev_depth_lss_r50_256x704_128x128_24e_2key.py` | bottom of file – changed experiment **name string** so log dir is unique (`'bev_depth_lss_r50_256x704_128x128_24e_2key'`) and *optionally* adjusted `data_root`, `info_train`, `info_val` to point at `/workspace/data/nuScenes_BEVDepth`. | makes experiment self‑contained inside Docker. |
| **No other algorithmic changes.** All kernels/ops are untouched – only dataset paths & logging names changed. |

> **Tip:** if you edit configs again, keep them next to originals (e.g. `*_2key_mydata.py`) to avoid future merges.

---

## 🧩 Library versions (frozen)

```
torch-1.12.1+cu116        mmcv-full-1.6.0
torchvision-0.13.1+cu116  mmdet-2.25.0
torchaudio-0.12.1+cu116   mmdet3d-1.0.0rc4
cuda-11.6 runtime         mmsegmentation-0.26.0
numba-0.56.4              llvmlite-0.39.1
```

All pinned in the Dockerfile & `requirements.txt`.  
Re-build = reproduce.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---------|-----|
| `AssertionError: MMCV==… incompatible` | You accidentally installed a different `mmcv-full`. Re-build or `pip install mmcv-full==1.6.0 -f …`. |
| `RuntimeError: cusparseCreate(handle)` | Torch & CUDA mismatch – run inside the provided image or rebuild after changing Torch. |
| `NameError: voxel_pooling_inference` | Re-compile ops: `python setup.py clean --all && python setup.py develop --no-deps`. |
| VS Code remote can’t see conda packages | Ensure you attached to **bevdepth-cu116** container; Python path is baked into the image. |

---

## 📄 Citation

If you use this environment, please cite **BEVDepth** (AAAI ’23):

```bibtex
@article{li2022bevdepth,
  title={BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection},
  author={Li, Yinhao and Ge, Zheng and ...},
  journal={arXiv preprint arXiv:2206.10092},
  year={2022}
}
```

---

*Happy BEV‑research 🚀 – open an issue/PR if you spot anything out of date!*
