FROM ubuntu:22.04

LABEL maintainer="Seth Parker <c.seth.parker@uky.edu>" \
      os="Ubuntu 22.04"

ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Copy the entire repository (this assumes Dockerfile is in the repository root)
COPY . /usr/local/educelab/pgs-recon

# Install apt dependencies and build photogrammetry dependencies
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
      build-essential \
      bzip2 \
      cmake \
      curl \
      doxygen \
      git \
      graphviz \
      imagemagick \
      libboost-iostreams-dev \
      libimage-exiftool-perl \
      libboost-program-options-dev \
      libboost-serialization-dev \
      libboost-system-dev \
      libboost-thread-dev \
      libbz2-dev \
      libgmp3-dev \
      libjpeg-turbo8-dev \
      libqt5opengl5-dev \
      libmpfr-dev \
      libqt5svg5-dev \
      libqt5x11extras5-dev \
      locales \
      nano \
      ninja-build \
      python3-dev \
      python3-pip \
      python3-venv \
      qtbase5-dev \
      qt5-qmake \
      software-properties-common \
      tzdata \
      wget \
      xz-utils \
      zlib1g-dev && \
    mkdir -p /usr/local/educelab/build/ && cd /usr/local/educelab/build/ && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_JPEG=OFF \
          -GNinja \
          /usr/local/educelab/pgs-recon/dependencies/ && \
    ninja && \
    locale-gen en_US en_US.UTF-8 && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    python3 -m venv /usr/local/educelab/pgs-recon/.venv && \
    . /usr/local/educelab/pgs-recon/.venv/bin/activate && \
    python3 -m pip install --upgrade pip wheel && \
    python3 -m pip install --editable /usr/local/educelab/pgs-recon && \
    chmod --recursive a+rw /usr/local/educelab/pgs-recon/ && \
    chmod a+rw /usr/local/lib/openMVG/sensor_width_camera_database.txt && \
    git config --global credential.helper "cache --timeout=3600" && \
    git config --global --add safe.directory /usr/local/educelab/pgs-recon/ && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
