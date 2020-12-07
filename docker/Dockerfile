FROM python:buster

# expect a build-time variable
ARG HDRTOOLS_VERSION=tags/v0.19.1
ARG HM_VERSION=tags/HM-16.22
ARG VTM_VERSION=tags/VTM-11.0 
ARG JM_VERSION=tags/JM-19.0

RUN DEBIAN_FRONTEND=noninteractive \
  apt update && apt install --assume-yes --no-install-recommends \
    build-essential wget ca-certificates unzip \
    cmake
  
WORKDIR /home/deps

## HDRtools ###
RUN git clone https://gitlab.com/standards/HDRTools.git && \
  git checkout ${HDRTOOLS_VERSION} && \ 
  mkdir -p ./HDRTools/build && cd HDRTools/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && \
  make -j

ENV HDRMETRICS_TOOL=/home/deps/HDRTools/build/bin/HDRMetrics

## HM ###
RUN git clone https://vcgit.hhi.fraunhofer.de/jct-vc/HM.git && \
  git checkout ${HM_VERSION} && \
  mkdir -p ./HM/build && cd ./HM/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && \
  make -j

ENV HM_ENCODER=/home/deps/HM/bin/TAppEncoderStatic
ENV HM_DECODER=/home/deps/HM/bin/TDecoderEncoderStatic

## VTM ###
RUN mkdir VTM && git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM ./VTM && \
  git checkout ${VTM_VERSION} \
  mkdir -p ./VTM/build && cd ./VTM/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && \
  make -j

ENV VTM_ENCODER=/home/deps/VTM/bin/EncoderAppStatic
ENV VTM_DECODER=/home/deps/VTM/bin/DecoderAppStatic

## JM ###
RUN git clone https://vcgit.hhi.fraunhofer.de/jct-vc/JM.git && \
  git checkout ${JM_VERSION} 

# ignoring warnings to build on linux here
COPY docker/JM.CMakeLists.txt JM/CMakeLists.txt

RUN mkdir -p JM/build && cd JM/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j

ENV JM_ENCODER=/home/deps/JM/bin/lencod_static
ENV JM_DECODER=/home/deps/JM/bin/ldecod_static

WORKDIR /home/5GVideo
COPY ./*.py ./
