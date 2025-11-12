FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# Set environment variables to avoid interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Set the CUDA architecture for PyTorch extension compilation
# This list covers most modern NVIDIA GPUs. Adjust if you have a specific architecture.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

WORKDIR /app

# Install system dependencies
 # added build-essential, cmake
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
# Create and activate virtual environment
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in virtual environment
RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# Install Segment Anything Model 
RUN pip3 install xformers==0.0.28
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git

# Install additional dependencies for SAM functionality
RUN pip3 install \
    opencv-python \
    pycocotools \
    matplotlib \
    onnxruntime \
    onnx \
    jupyter \
    numpy \
    Pillow
# installing unidepth libraries 
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir --no-binary mmcv

# installing dino 
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO && \
    rm -rf build/ dist/ *.egg-info \
    pip install -e . --no-build-isolation

# COPY install_groundingdino.sh .
# RUN bash install_groundingdino.sh

RUN pip3 install pydantic==2.12.3
