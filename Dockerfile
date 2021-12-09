FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
        wget \
        nginx \
        ca-certificates \
        tmux \
        mc \ 
        nano \
        build-essential \
        rsync \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html && \
    pip install numpy scipy opencv-python tensorflow joblib matplotlib pandas \
    albumentations==0.5.2 pytorch-lightning==1.2.9 tabulate easydict==1.9.0 kornia==0.5.0 webdataset \
    packaging gpustat tqdm pyyaml hydra-core==1.1.0.dev6 scikit-learn==0.24.2 tabulate scikit-image==0.17.2 \
    gunicorn==19.9.0 gevent flask && \
        rm -rf /root/.cache

ENV TORCH_HOME="/home/$USERNAME/.torch"

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"


COPY ./image_module /opt/program
WORKDIR /opt/program