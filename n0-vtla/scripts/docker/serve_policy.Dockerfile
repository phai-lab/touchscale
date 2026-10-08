# Dockerfile for serving a policy.

# Build the container:
# docker build . -t n0vtla_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all n0vtla_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        curl \
        git \
        git-lfs \
        linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*

ARG MINICONDA_VERSION=py311_25.5.1-0
RUN curl -fsSL \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
        -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda create -y -n vtla python=3.11 \
    && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/envs/vtla/bin:/opt/conda/bin:$PATH
ENV PYTHONPATH=/app

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    GIT_LFS_SKIP_SMUDGE=1 python -m pip install -r /tmp/requirements.txt

# Copy transformers_replace files while preserving directory structure
COPY n0vtla/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

CMD ["/bin/bash", "-c", "python scripts/serve_policy.py $SERVER_ARGS"]
