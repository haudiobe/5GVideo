FROM python:buster

ARG HM_VERSION=tags/HM-16.22
ARG SCM_VERSION=tags/HM-16.21+SCM-8.8
ARG VTM_VERSION=tags/VTM-11.0
# adds CMakelist.txt on top of tags/JM-19.0
ARG JM_VERSION=7901703651acf5a64df55615d02de261e9c0ee87
ARG HDRTOOLS_VERSION=tags/v0.19.1
ARG VMAF_VERSION=tags/v2.1.1
# build HDRTOOLS without JM_PSNR C flag
ARG jm_psnr_undef

RUN DEBIAN_FRONTEND=noninteractive \
  apt update && apt install --assume-yes --no-install-recommends \
    build-essential wget ca-certificates unzip \
    cmake

WORKDIR /home/deps

## HDRtools ###
RUN git clone https://gitlab.com/standards/HDRTools.git && \
  cd ./HDRTools && git checkout ${HDRTOOLS_VERSION}

# docker build --build-arg jm_psnr_undef=1 to use this option
COPY docker/HDRTOOLS_disable_jm_psnr.patch ./HDRTools/HDRTOOLS_disable_jm_psnr.patch
RUN [ ! -z "${jm_psnr_undef}" ] && cd ./HDRTools && git apply ./HDRTOOLS_disable_jm_psnr.patch || echo "build HDRTOOLS w/ -D JM_PSNR"

RUN mkdir -p ./HDRTools/build && cd ./HDRTools/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && \
  make -j 4

ENV HDRMETRICS_TOOL=/home/home/deps/HDRTools/build/bin/HDRMetrics
ENV HDRTOOLS_CHOMACONVERT=/home/deps/HDRTools/build/bin/ChromaConvert

## HM ###
RUN git clone https://vcgit.hhi.fraunhofer.de/jct-vc/HM.git && \
  cd ./HM && git checkout ${HM_VERSION}
RUN mkdir -p ./HM/build && cd ./HM/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && \
  make -j 4

ENV HM_ENCODER=/home/deps/HM/bin/TAppEncoderStatic
ENV HM_DECODER=/home/deps/HM/bin/TAppDecoderStatic

## VTM ###
RUN mkdir VTM && git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM ./VTM && \
  cd ./VTM && git checkout ${VTM_VERSION}

RUN mkdir -p ./VTM/build && cd ./VTM/build && \
  ln -s /home/deps/HDRTools/common /home/deps/VTM/source/Lib/HDRLib && \
  cmake .. -DCMAKE_BUILD_TYPE=Release -DEXTENSION_HDRTOOLS=on && \
  make -j 4

ENV VTM_ENCODER=/home/deps/VTM/bin/EncoderAppStatic
ENV VTM_DECODER=/home/deps/VTM/bin/DecoderAppStatic

## JM ###
RUN git clone https://vcgit.hhi.fraunhofer.de/jct-vc/JM.git && \
  cd JM && git checkout ${JM_VERSION} 

# ignoring warnings to build on linux here
COPY docker/JM_ignore_compiler_warnings.patch ./JM/JM_ignore_compiler_warnings.patch
RUN cd ./JM && git apply ./JM_ignore_compiler_warnings.patch

RUN mkdir -p ./JM/build && cd JM/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j 4

ENV JM_ENCODER=/home/deps/JM/bin/lencod_static
ENV JM_DECODER=/home/deps/JM/bin/ldecod_static

## VMAF ###
RUN apt install --assume-yes --no-install-recommends nasm ninja-build doxygen && \
  pip install --upgrade pip && pip install --no-cache-dir meson cython numpy
RUN git clone https://github.com/Netflix/vmaf.git && \
  cd ./vmaf && git checkout ${VMAF_VERSION} && \
  make
ENV PATH=/home/deps/vmaf:/home/deps/vmaf/libvmaf/build/tools:$PATH
ENV VMAF_MODEL=path=/home/deps/vmaf/model/vmaf_v0.6.1.json:enable_transform
ENV VMAF_EXE=/home/deps/vmaf/libvmaf/build/tools/vmaf

## HM SCC extension ###
RUN cp -r /home/deps/HM /home/deps/SCM
RUN cd ./SCM && rm -rf ./bin/* && rm -rf ./build/* && git checkout ${SCM_VERSION}
RUN cd ./SCM/build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release && \
  make -j 4
ENV SCM_ENCODER=/home/deps/SCM/bin/TAppEncoderStatic
ENV SCM_DECODER=/home/deps/SCM/bin/TAppDecoderStatic

## ETM public repo ? ###

ENV SEI_REMOVAL_APP=/home/deps/HM/bin/SEIRemovalAppStatic

WORKDIR /home/5GVideo
COPY requirements.txt ./
RUN pip install -r ./requirements.txt
COPY *.py ./
COPY report-template.json ./

# ENTRYPOINT ["/usr/bin/python3"]