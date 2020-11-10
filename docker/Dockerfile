FROM python:buster

RUN DEBIAN_FRONTEND=noninteractive \
  apt update && apt install --assume-yes --no-install-recommends \
    build-essential wget ca-certificates unzip \
    # FFMPEG dependencies
    nasm \
    # GPAC dependencies
    zlib1g-dev \
    git \
    # JCT-VT SW
    cmake
  
WORKDIR /home/deps

# build libav / ffmpeg
RUN wget https://ffmpeg.org/releases/ffmpeg-4.3.1.tar.bz2 \
    && tar xfj ffmpeg-4.3.1.tar.bz2 \
    && cd ffmpeg-4.3.1 \
    && ./configure --enable-shared --disable-static\
    && make -j \
    && make install

# build gpac from master until 1.1.0 is tagged for YUV muxed to .mp4
RUN git clone https://github.com/gpac/gpac.git && \
  cd gpac && \
  ./configure && \
  make && \
  make install

## HM ###
RUN git clone https://vcgit.hhi.fraunhofer.de/jct-vc/HM.git && \
  mkdir -p ./HM/build && cd ./HM/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && \
  make -j

ENV HM_ENCODER=/home/deps/HM/bin/TAppEncoderStatic
ENV HM_DECODER=/home/deps/HM/bin/TDecoderEncoderStatic

## VTM ###
RUN mkdir VTM && git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM ./VTM && \
  mkdir -p ./VTM/build && cd ./VTM/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && \
  make -j

ENV VTM_ENCODER=/home/deps/VTM/bin/EncoderAppStatic
ENV VTM_DECODER=/home/deps/VTM/bin/DecoderAppStatic

## JM ###
RUN git clone https://vcgit.hhi.fraunhofer.de/jct-vc/JM.git

# @FIXME: ignoring warnings to build on linux here
COPY docker/JM.CMakeLists.txt JM/CMakeLists.txt

RUN mkdir -p JM/build && cd JM/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j

ENV JM_ENCODER=/home/deps/JM/bin/lencod_static
ENV JM_DECODER=/home/deps/JM/bin/ldecod_static

WORKDIR /home/5GVideo
COPY ./*.py ./
