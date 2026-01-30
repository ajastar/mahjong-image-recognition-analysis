# NVIDIA公式のCUDAとPythonが含まれたベースイメージを使用
# JAXが推奨するPythonバージョンと互換性のあるものを選択
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 環境設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Tokyo

# JAX、PGX、Tensorflow（警告抑制用）などのライブラリをインストール
# jaxlibのインストール時に、CUDAのバージョン（11.8）を指定してビルド済みバージョンを確実にインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# jaxlibのCUDA 11.8対応版を指定してインストール
# この行が、これまでの全ての警告とエラーを解決します
RUN pip3 install --upgrade pip
RUN pip3 install jax[cuda11_pip]==0.4.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# その他の必要なライブラリをインストール
RUN pip3 install numpy pgx optax flax absl-py tensorflow-cpu

# アプリケーションコードのコピー
WORKDIR /app
COPY . /app

# エントリポイントはそのまま
CMD ["/bin/bash"]