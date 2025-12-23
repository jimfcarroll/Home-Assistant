FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OLLAMA_MODELS=/models/ollama

# Install dependencies
RUN apt update && apt install -y \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    pciutils \
    lshw \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# Expose Ollama API port
EXPOSE 11434

# Start Ollama server
CMD ["ollama", "serve"]
