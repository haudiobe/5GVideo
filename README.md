
# Overview

1. build / environment
2. downloading reference content from the [content server](https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/)
3. video conversion step (8 to 10bit, EXR)
4. creating test data
5. running verifications

```
**.py --scenario_dir /data/Bitstreams/{scenario}/{test} -k {anchor} [--dry-run]
```

the scripts process the target anchor in the target scenario.
- anchor must be listed in `/data/Bitstreams/{scenario}/{test}/streams.csv`
- the reference sequence for the anchor must be listed in `/data/Bitstreams/{scenario}/reference-sequences.csv`

_____

# 1. build / environment

## 1.1 Docker image

The sample [Dockerfile](https://docs.docker.com/get-docker/) located in the `docker/` directory, aims at bundling the scripts and dependencies in a portable reference environment: JM, HM, VTM, SCM, VMAF, HDRTools.

**build the docker image**

When building the docker image, software version can be added though docker --build-args. 
This is usefull to create docker images for scpecific scenarios.

```
git clone https://github.com/haudiobe/5GVideo.git
cd 5GVideo

# building an image for Scenario 3
docker build \
 --build-arg JM_VERSION=master \
 --build-arg HM_VERSION=tags/HM-16.22 \
 --build-arg SCM_VERSION=tags/HM-16.21+SCM-8.8 \
 --build-arg VTM_VERSION=tags/VTM-11.0 \
 --build-arg HDRTOOLS_VERSION=tags/v0.22 \
 --build-arg VMAF_VERSION=tags/v2.1.1 \
 -t anchortools:Scenario-3-Screen -f ./docker/Dockerfile .
```

The 'version' variables *must be valid git tags* used to checkout the source repositories. Note that, in the case of JM, the master branch yields JM v19 source plus build toolchain updates that are not available in the v19 tag.


**using the docker image**

to run the scripts in docker container, you can mount your local data directory to `/data`, then specify the script and options.

```
base_dir=/path/to/host/data
docker run -it \
    --mount type=bind,source=$base_dir,target=/data anchortools:latest \
    python3 ./verify.py decoder --scenario_dir /data/Bitstreams/Scenario-3-Scene -k S3-A36-265
```


## 1.2 running the scripts without docker

```
git clone https://github.com/haudiobe/5GVideo.git
cd 5GVideo
```

use a python virtual environment
```
python3 -m venv .venv 
source .venv/bin/activate
```

install the dependencies:
```
pip3 install -r requiements.txt
```

Configure environment variables with your local executable path:
```
HM_ENCODER=/path/to/HM/bin/TAppEncoderStatic
HM_DECODER=/path/to/HM/bin/TAppDecoderStatic
SCM_ENCODER=/path/to/SCM/bin/TAppEncoderStatic
SCM_DECODER=/path/to/SCM/bin/TAppDecoderStatic
VTM_ENCODER=/path/to/VVCSoftware_VTM/bin/EncoderAppStatic
VTM_DECODER=/path/to/VVCSoftware_VTM/bin/DecoderAppStatic
JM_ENCODER=/path/to/JM/bin/lencod_static
JM_DECODER=/path/to/JM/bin/ldecod_static
ETM_ENCODER=/path/to/ETM-master/bin/evca_encoder
ETM_DECODER=/path/to/ETM-master/bin/evca_decoder
SEI_REMOVAL_APP=/path/to/HM/bin/SEIRemovalAppStaticd
HDRMETRICS_TOOL=/path/to/HDRTools/build/bin/HDRMetrics
HDRCONVERT_TOOL=/path/to/HDRTools/build/bin/HDRConvert
VMAF_EXEC=/path/to/vmaf/libvmaf/build/tools/vmaf
```

Make sure these point to the correct software version for each scenario.

_____


# 2. downloading content from the reference server

The `download.py` script downloads content from the public [content server](https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/).

When content is already available in the target directory, it is skipped if the local file size matches the original file.

`--dry-run` doesn't proceed with download, but instead it lists files that need to be downloaded.

**downloading reference sequences needed for a given scenario/codec**

```
download.py --dl-ref-sequences --scenario_dir /data/Bitstreams/Scenario-3/265-Screen/H265
```

this download the `reference-sequence.csv` for the target scenario 'Scenario-3-Screen',
reads that csv, and proceeds with downloading all the reference sequences (the sidecar metadata file, and video sequence).


**downloading all variants of a given scenario/codec**

```
download.py --dl-streams --scenario_dir /data/Bitstreams/Scenario-3-Screen/265
```

downloads the encoder configuration, and all variant data

_____


# 3. content conversion

For some scenario/metric configurations it is required to pre-process content before running HDR tools (eg. 8 bit ref sequence with 10bit coded bit depth). To generate the required conversions:
```
convert.py --scenario_dir /data/Bitstreams/Scenario-3-Screen/265 -k S3-A36-265
```
If a conversion already exists (.json + .yuv both exist) it will be used, otherwise a new one will be generated.


_____


# 5. creating new test data

the scripts can be used to generate new data for a given **scenario** / **encoder** :
- `encoder`: encodes streams based on *Bitstreams/scenario/codec/streams.csv* and *Bitstreams/scenario/reference-sequences.csv*
- `decoder`: decode stream, compute metrics and update individual streams json metadata
- `metrics`: compute metrics and update individual streams json metadata


## 5.1 anchor bitstreams generation

encode a specific anchor:
```
create.py encoder --scenario_dir /data/Bitstreams/Scenario-3/265 -k S3-A36-265 encoder
```

encode an entire scenario/codec (sequentialy):
```
create.py encoder --scenario_dir /data/Bitstreams/Scenario-3/265
```


## 5.2 metrics generation

> After bitstreams generation, and **before running metrics generation**, make sure to run `convert.py`.

*run decoding and compute metrics* for a specific anchor:
```
create.py decoder --scenario_dir /data/Bitstreams/Scenario-3/265 -k S3-A36-265
```

*compute metrics without decoding step* for a specific anchor, ensuring if it was already properly reconstructed:
```
create.py metrics --scenario_dir /data/Bitstreams/Scenario-3/265 -k S3-A36-265
```

to process the entire scenario/codec, remove `-k S3-A36-265` from the above commands


### 5.2.1 metrics softwares

**PSNR / MS-SSIM**

HDRTools is used for metrics computation. It is built with default flags, in particular `-D JM_PSNR` compilation flag is set.

**Bitstream size / bitrate**

For HM / SCM:
- the `BitrateLog` metric is parsed from encoder log.
- the `Bitrate` metric is computed based on the file size and expected to match SEI are removed from bitstream 

**VMAF**

the VMAF executable used is : `libvmaf/build/tools/vmaf`
the vmaf model used is configured through environment variable, eg.: 
    `VMAF_MODEL=path=/home/deps/vmaf/model/vmaf_v0.6.1.json:enable_transform`

when running the docker image, this can be customized easily using docker run's `--env` options, eg:
```
base_dir=/path/to/host/data
docker run -it \
    --mount type=bind,source=$base_dir,target=/data anchortools:latest \
    --env VMAF_MODEL=/home/deps/vmaf/model/vmaf_v0.6.1.json:enable_transform \
    python3 ./verify.py decoder --scenario_dir /data/Bitstreams/Scenario-3-Scene -k S3-A36-265
```


**Decoder/Encoder log metrics**

Decoder / Encoder can implement parsing metrics from logs. Currently, only HM and SCM implement log parsing.


## 5.3 encoder/decoder implementation

To implement new encoders, take look at `encoders.py`. 
You need to subclass the EncoderBase class and decorate your class (@register_encoder).
Encoder implementation only need to supply an encoder ID, 
implement the function that generate the shell command lines for encoding/decoding,
and optionaly parse metrics from the logs or stat files they produce.

_____


# 5. content verification

`verify.py` script runs verification for bitstream or metrics, and updates the anchor's vairant bitstream json with a new verification status.


**bitstream verification**

for a specific anchor:
```
verify.py --scenario_dir /data/Bitstreams/Scenario-3/265 -k S3-A36-265 bitstream 
```

for the entire scenario/codec:
```
verify.py --scenario_dir /data/Bitstreams/Scenario-3/265 bitstream 
```

**decoder verification**

for a specific anchor:
```
verify.py --scenario_dir /data/Bitstreams/Scenario-3/265 -k S3-A36-265 decoder
```

for the entire scenario/codec:
```
verify.py --scenario_dir /data/Bitstreams/Scenario-3/265 decoder
```


## bundling verification reports to csv

When running verification steps, the result is stored directly in the anchor json. To export the most recent verification report to csv, use the following commands :
```
verify.py report --scenario_dir /data/Bitstreams/Scenario-3/265 \
    -k S3-A36-265 \
    --template ./report-template.json
```
to generate `/data/Bitstreams/Scenario-3/265/verification_report.csv`.

the `--template` argument specifies a json template for the report (contact info, etc ...).

sample `./report-template.json` :
```
{
    "Contact": {
        "Company": "Co",
        "name": "Name",
        "e-mail": "e@mail.me"
    },
    "meeting": "",
    "input": ""
}
```

_____

# FAQ

> I have my own encoding scripts, how do I feed the results ?

These scripts should support your workflow providing that:
- your data follows the reference directory layout
- you can generate the bitstream json metadata to describe and point to your bistream file.

Your `Bitstreams/scenario/test/streams.csv` must list all the anchors for your test on a given scenario. The anchors/variants should be located in that directory too. Each CSV row describes an anchor, which maps to a subfolder containing its variants. 

The scripts load the bistream json metadata for each variant, providing all the informations to perform post-encoding processing steps.

Take a look at `anchors.py`, it provides the `VariantData` which loads/saves the individual bitstream's json metadata. it also provides functions to iterate the csv files.