FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

WORKDIR /working

ENV LDFLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"

RUN apt update \
    && apt-get install -y git build-essential cmake ccache libcurl4-openssl-dev \
    && echo "==========================================" \
    && echo "NVCC:" \
    && nvcc --version \
    && echo "==========================================" \
    && git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
       -DCMAKE_BUILD_TYPE=Release \
       -DGGML_CUDA=ON \
       -DGGML_CUDA_FA_ALL_QUANTS=ON \
       -DGGML_IQK_FA_ALL_QUANTS=ON \
       -DGGML_CUDA_IQK_FORCE_BF16=1 \
       -DGGML_BLAS=OFF \
       -DGGML_SCHED_MAX_COPIES=1 \
       -DCMAKE_CUDA_ARCHITECTURES=120 \
    && cmake --build build --config Release -j $(nproc)



