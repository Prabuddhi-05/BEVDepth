###############################################################################
# BEVDepth – CUDA 11.6 / Torch 1.12.1 / OpenMMLab 1.x stack
###############################################################################
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# ─────────────────────────────  APT  ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl wget build-essential \
        libglib2.0-0 libsm6 libxext6 libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

# ───────────────────────────  Miniconda  ──────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p $CONDA_DIR \
 && rm /tmp/mc.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda clean -afy

# Later:
ENV PYTHONPATH="/workspace/BEVDepth:${PYTHONPATH}"

# ─────────────────────────  Python core + Torch  ─────────────────────────────
RUN conda install -y python=3.8 pip && conda clean -afy

# Torch 1.12.1 / cu116 wheels
RUN pip install --no-cache-dir \
      torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 \
      -f https://download.pytorch.org/whl/torch_stable.html

# ───────────────────────────  OpenMMLab stack  ───────────────────────────────
# mmcv-full 1.6.0 compiled for Torch 1.12/cu116
RUN pip install --no-cache-dir \
      mmcv-full==1.6.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html

# matching detection / segmentation libs
RUN pip install --no-cache-dir \
      mmdet==2.25.0 \
      mmdet3d==1.0.0rc4 \
      mmsegmentation==0.26.0

# ───────────────────────  “generic” requirements  ────────────────────────────
COPY requirements.txt /tmp/requirements.txt
COPY constraints.txt  /tmp/constraints.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/constraints.txt

# ─────────────────────────────  Source  ──────────────────────────────────────
WORKDIR /workspace/BEVDepth

# ────────────────────────────  Entrypoint  ───────────────────────────────────
# These two env-vars make life easier inside the container
ENV PYTHONPATH=/workspace/BEVDepth:$PYTHONPATH
ENV FORCE_CUDA=1

WORKDIR /workspace/BEVDepth
CMD ["/bin/bash"]
