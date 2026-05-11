FROM docker-hosted.artifactory.tcsbank.ru/tfusion-ml/ai-toolkit:v0.0.2

ARG DEBIAN_FRONTEND=noninteractive

# Удаляем заложенные репозитории
RUN rm -rf /etc/apt/sources.list.d/*

# Добавляем проксирующие репозитории
RUN echo "deb http://repo-linux.tcsbank.ru/ubuntu/ focal multiverse main universe restricted" > /etc/apt/sources.list && \
    echo "deb http://repo-linux.tcsbank.ru/ubuntu/ focal-backports multiverse main universe restricted" >> /etc/apt/sources.list && \
    echo "deb http://repo-linux.tcsbank.ru/ubuntu/ focal-proposed multiverse main universe restricted" >> /etc/apt/sources.list && \
    echo "deb http://repo-linux.tcsbank.ru/ubuntu/ focal-security multiverse main universe restricted" >> /etc/apt/sources.list && \
    echo "deb http://repo-linux.tcsbank.ru/ubuntu/ focal-updates multiverse main universe restricted" >> /etc/apt/sources.list

ENV PIP_INDEX_URL=https://artifactory.tcsbank.ru/artifactory/api/pypi/python-all/simple
ENV TRANSFORMERS_CACHE="/root/huggingface_cache"
ENV HF_HOME="/root/huggingface_cache"
ENV IS_METRICS_NEEDED=0
ENV USE_CLEARML=0

WORKDIR /root
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    s3cmd \
    sudo \
    git \
    wget \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel

WORKDIR /root/b-lora-flux
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock /root/b-lora-flux/
RUN poetry config virtualenvs.create false
RUN poetry lock
RUN poetry install --no-root -v

ENV PATH="$HOME/.local/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/root/"

# WORKDIR /root/
# RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

USER ${USERNAME}
