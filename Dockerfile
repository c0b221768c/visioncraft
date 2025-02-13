FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

RUN apt update && apt install -y \
    x11-apps \
    libgstreamer1.0-dev gstreamer1.0-tools \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-plugins-base \
    kmod \
    google-perftools \
    cmake \
    libopenblas-dev libomp-dev libgtest-dev \
    swig

WORKDIR /root
RUN git clone --depth 1 --branch v1.7.4 https://github.com/facebookresearch/faiss.git

WORKDIR /root/faiss
RUN mkdir build && cd build && \
    cmake -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    make install

# Python バインディングをインストール
RUN cd /root/faiss/build && \
    pip install faiss/python insightface retina-face

# Python パッケージの整理
RUN pip cache purge

RUN export PYTHONPATH=/app/
