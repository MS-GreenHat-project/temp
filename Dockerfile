# 1. AzureML 공식 베이스 이미지 (Python 3.8, Ubuntu 20.04)
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# 2. 시스템 패키지(apt) 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# 3. conda 패키지 설치 (xorg 관련)
RUN conda install -y \
        xorg-libxrender \
        xorg-libxext \
        xorg-libsm \
        xorg-libice

# 4. pip 업그레이드 및 패키지 설치
RUN pip install --upgrade pip
RUN pip install \
    azureml-core \
    azureml-pipeline-core \
    azureml-pipeline-steps \
    azureml-mlflow \
    "azureml-dataprep[fuse,pandas]" \
    ultralytics \
    mlflow \
    pyyaml \
    pandas \
    azureml-defaults \
    tqdm \
    scikit-learn \
    matplotlib \
    seaborn \
    numpy \
    opencv-python-headless

WORKDIR /greenhat-ai

# 6. 환경 변수
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg    