# video codec charcterization scripts

these script can be used to produce / verify specific encoders and scenarios according to TR26.955.

- [video codec charcterization scripts](#video-codec-charcterization-scripts)
  - [installation](#installation)
    - [working directory](#working-directory)
    - [download content from server](#download-content-from-server)
  - [vcc.py script usage](#vccpy-script-usage)
    - [processing all QPs of a an anchor](#processing-all-qps-of-a-an-anchor)
    - [processing a single QP for an anchor](#processing-a-single-qp-for-an-anchor)
    - [process all anchors for a given encoder config](#process-all-anchors-for-a-given-encoder-config)
    - [`encode`, `decode`, `convert`, `metrics` subcommands](#encode-decode-convert-metrics-subcommands)
      - [common options](#common-options)
      - [encode](#encode)
      - [decode](#decode)
      - [convert](#convert)
      - [metrics](#metrics)
        - [image quality metrics](#image-quality-metrics)
        - [bitrate / bistream size](#bitrate--bistream-size)
        - [metrics csv export](#metrics-csv-export)
  - [characterization, rd curves plots, bd rate](#characterization-rd-curves-plots-bd-rate)
  - [using the scripts for metrics verification](#using-the-scripts-for-metrics-verification)
    - [working directory](#working-directory-1)
    - [download bitstreams for verification](#download-bitstreams-for-verification)
    - [reconstruction and metrics](#reconstruction-and-metrics)
      - [generate reconstruction of bitstreams](#generate-reconstruction-of-bitstreams)
      - [format conversion before metrics computation](#format-conversion-before-metrics-computation)
      - [compute metrics](#compute-metrics)
        - [export metrics from .json to .csv](#export-metrics-from-json-to-csv)
    - [generate verification report](#generate-verification-report)
      - [download the original metrics](#download-the-original-metrics)
      - [metrics verification](#metrics-verification)
  - [docker usage](#docker-usage)
    - [building the docker images](#building-the-docker-images)
    - [using the worker image](#using-the-worker-image)
    - [using docker-compose](#using-docker-compose)


## installation

```
git clone https://github.com/haudiobe/5GVideo 
cd 5GVideo
```

Please use a [python virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) to install dependencies and run the scripts. 

install python dependencies:
```
pip3 install -r src/requirements.txt
```


### working directory

The scripts assume you have a local directory where content has been downloaded from: https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/
or follows that structure. 

No default value is provided for the path to the working directory, it must be supplied explicitely by one of these 2 means:
- the scripts accept a `--working-dir` argument to supply the path.
- the `VCC_WORKING_DIR` environment variable is used to configure the location of that directory


### download content from server

to download anchors, bitstreams, metrics, encoder configs, & reference sequence list for a given scenario:
```
python3 src/download.py streams https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/Bitstreams/Scenario-1-FHD/264/streams.csv $VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/264
```

to download anchors reference sequences for a csv list:
```
python3 src/download.py sequences $VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/reference-sequences.csv $VCC_WORKING_DIR/ReferenceSequences
```



## vcc.py script usage

`src/vcc.py` provides the commands to encode, decode, and compute metrics.

All the commands described below have a common usage pattern.


### processing all QPs of a an anchor

```
python3 src/vcc.py [--dry-run] -s S1-A01-264 decode
```

the scripts parses `-s S1-A01-264`, finds `Scenario-1-FHD/265/streams.csv`, and processes **all QPs** listed for `S1-A01-264`.

### processing a single QP for an anchor

```
python3 src/vcc.py [--dry-run] -s S1-A01-264-22 decode
```
the scripts works as described above, but will only process a **single QP** - in this case: **22**.


### process all anchors for a given encoder config

```
python3 src/metrics.py [--dry-run] -c S1-JM-01 decode
```

the scripts parses `-c S1-JM-01`, finds `Scenario-1-FHD/264/streams.csv`, then processes **all QPs,** for **all anchors** where the encoder config is `S1-JM-01`.



### `encode`, `decode`, `convert`, `metrics` subcommands

#### common options

* `--dry-run`: does not execute the subprocesses, but instead prints the corresponding commands to stdout.
* `--queue`: advanced - pushes the commands as tasks to a worker queue using the celery framework, instead of processing sequentialy. This option is meant to be used with the `sample docker-compose.yml` stack. See *Docker usage* > *Using docker-compose*.


#### encode

```
JM19_0_ENCODER=~/deps/JM/bin/lencod_static

python3 src/vcc.py -s S1-A01-264 encode
```

*dry-run* mode prints the command used to encode to stdout but doesn't run it, but it does compute existing bitstream md5.


#### decode

```
JM19_0_DECODER=~/deps/JM/bin/ldecod_static

python3 src/vcc.py -s S1-A01-264 decode
```

*dry-run* mode prints the command used to encode to stdout but doesn't run it, but it does compute existing reconstruction md5.


#### convert

```
HDRCONVERT_TOOL=~/deps/HDRTools/build/bin/HDRConvert

python3 src/vcc.py -s S1-A01-264 decode --reference
python3 src/vcc.py -s S1-A01-264 decode --reconstructions
```

Performs conversion needed before metrics computation using HDRTools' HDRConvert.

when --reconstructions is specfied, the reconstructions for all QPs are processed. 
when --reference is specified, the reference sequence is processed instead.

Conversion is either be 8 to 10 bit YUV, or YUV to EXR. **When conversion is not needed for a given sequence, the script does nothing.**

The converted sequences are stored in a tmp subfolder for each anchor.

*dry-run* mode prints the command used to convert to stdout but doesn't run it.


#### metrics

when running the metrics command, the result is stored in the json file.

##### image quality metrics

```
HDRMETRICS_TOOL=~/deps/HDRTools/build/bin/HDRMetrics
VMAF_EXE=~/deps/vmaf/libvmaf/build/tools/vmafossexec
VMAF_MODEL=/home/deps/vmaf/model/vmaf_v0.6.1.json

python3 src/vcc.py -s S1-A01-264 metrics
```

Computes metrics for the given sequence. This step also computes bitrate after removing SEI.

HDRTOOLS or VMAF can be enabled / disabled through environment variables.
```
DISABLE_HDRMETRICS=1
DISABLE_VMAF=1
```

This step is meant to run after successfull `conversion`. It uses the converted sequence where appropriate, and raise an error if it doesn't exist.

*dry-run* mode doesn't run HDRTools or vmaf, but it parses existing logs if any.

##### bitrate / bistream size

Bitstream size / bitrate is always computed when running metrics. It can be computed explicitly as a separate step.
```
python3 src/vcc.py -s S1-A01-264-22 bitrate
```
First, SEI_REMOVAL_APP is used if implemented by the encoder class.
Then bitrate is computed based on the resulting bitstream size, and the json metadata are updated with the result.


##### metrics csv export

Metrics are saved on the anchors JSON file. A script can be used to export them to .csv:

```
python3 src/metrics.py -s S1-A01-264 csv-metrics
python3 src/metrics.py -c S1-HM-02 csv-metrics
```

For the above commands, the metrics `.csv` files can then be found in: `$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/265/Metrics`




## characterization, rd curves plots, bd rate

The characterization script is meant to compare test data to an anchor. For instance, to compare sequences `JM` and `HM` for `Scenario-1-FHD`, you should have the following data:

```
$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/
$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/264/S1-A01-264.json
$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/264/streams.csv
$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/264/Metrics/S1-A01-264.csv
[...]

$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/
$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/264/S1-A01-265.json
$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/265/streams.csv
$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/265/Metrics/S1-A01-264.csv
[...]
```


To compare and plot psnr, y_psnr, and ms_ssim for these 2 sequences: 
```
python3 src/compare.py --plot -s S1-A01-264 S1-A01-265 psnr y_psnr ms_ssim
```

Assuming you have all the anchors for encoder config S1-JM-01 and S1-HM-01, you can process them at once:
```
python3 src/compare.py --plot -c S1-JM-01 S1-HM-01 psnr y_psnr ms_ssim
```


_____


## using the scripts for metrics verification

This guide provides step by step instructions to process with reconstruction and metrics verifications.

### working directory

In this guide, `/data` will be the directory where reference content is downloaded.
All subsequent verification steps assume the `VCC_WORKING_DIR` environment variable is set to `/data`.


### download bitstreams for verification

1. download bitstreams and metadata for a given scenario

```
python3 src/download.py streams --bitstream https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/Bitstreams/Scenario-1-FHD/VTM/streams.csv /data/Bitstreams/Scenario-1-FHD/VTM


/data
 └ Bitstreams
  └ Scenario-1-FHD
    |   reference-sequence.csv
    └ VTM
     |   streams.csv
     └── CFG
     └── Metrics
     └── S1-T11-VTM
     |   └── S1-T11-VTM-22
     |       S1-T11-VTM-22.json
     |       S1-T11-VTM-22.bin
     |   └── S1-T11-VTM-27
     |   └── S1-T11-VTM-32
     |   └── S1-T11-VTM-37
    [...]
     └ S1-T17-VTM
```


2. download reference sequences for that scenario

```
python3 src/download.py sequences /data/Bitstreams/Scenario-1-FHD/reference-sequence.csv /data/ReferenceSequences


/data
  └ ReferenceSequences
    └ Life-Untouched
        Life-Untouched-FHD.json
        Life-Untouched-FHD.yuv
```

### reconstruction and metrics

#### generate reconstruction of bitstreams

reconstruct QP 22 for S1-T11-VTM
```
python3 src/vcc.py decode -s S1-T11-VTM-22 decode


/data/Bitstreams/Scenario-1-FHD
 └── S1-T11-VTM
     └── S1-T11-VTM-22
         S1-T11-VTM-22.yuv
         S1-T11-VTM-22.dec.log
```


#### format conversion before metrics computation

In some cases, must process conversion from one representation to another before.
This 8 to 10 bit, 10 to 8 bit, and YUV to RGB444.

In the case of S1-T11-VTM, YUV to RGB444 needs to be performed on the reference sequence as well as on the reconstructions to compute HDR metrics using HDRTools.


Convert the reconstructions of the test scenario:

```
python3 src/vcc.py -s S1-T11-VTM-22 convert --reconstructions


/data/Bitstreams/Scenario-1-FHD
 └── S1-T11-VTM
     └── S1-T11-VTM-22
         |   S1-T11-VTM-22.yuv
         |   S1-T11-VTM-22.yuv.json
         └── tmp
                S1-T11-VTM-22_2020_444_%05d.json
                S1-T11-VTM-22_2020_444_000000.exr
                S1-T11-VTM-22_2020_444_000000.exr
                S1-T11-VTM-22_2020_444_000001.exr
                ...
                S1-T11-VTM-22_2020_444_%05d.hdrconvert.log
     ...
```

Convert the reference sequence used to encode S1-T11-VTM:

```
python3 src/vcc.py -s S1-T11-VTM convert --sequences


/data/ReferenceSequences
 └── Life-Untouched
     |   Life-Untouched-FHD.yuv
     |   Life-Untouched-FHD.json
     └── tmp
            Life-Untouched-FHD_2020_444_%05d.json
            Life-Untouched-FHD_2020_444_000000.exr
            Life-Untouched-FHD_2020_444_000001.exr
            ...
            Life-Untouched-FHD_2020_444_%05d.hdrconvert.log
 ...
```


#### compute metrics

Run metrics computations, and updates the json metadata.

```
python3 src/vcc.py -s S1-T11-VTM-22 metrics

/data/Bitstreams/Scenario-1-FHD
 └── S1-T11-VTM
     └── S1-T11-VTM-22
         |   S1-T11-VTM-22.metrics.log
         |   S1-T11-VTM-22.json
         |   ...
         └── tmp
                S1-T11-VTM-22_2020_444_%05d.metrics.log
                ...
```

##### export metrics from .json to .csv

When computing metrics, results are stored in json metadata. After computing metrics for all anchors of a given encoder configuration, the metrics are exported to the appropriate csv format. 

this is conveniently done for a set of all anchors with a given encoder config: 
```
python3 src/metrics.py -c S1-VTM-O2 csv-metrics

/data/Bitstreams/Scenario-1-FHD/Metrics
 └── ...
     S1-VTM-02.csv
```

but can also be done for individual anchor:
```
python3 src/metrics.py -s S1-T11-VTM csv-metrics
/data/Bitstreams/Scenario-1-FHD/Metrics
 └── S1-T11-VTM.csv
```

### generate verification report

to generate a verification report, we compare the local reconstruction and metrics computed in the working directory, to the original data downloaded from the server.


#### download the original metrics

create the directory `/data/Origin/Scenario-1-FHD/VTM` and download the reference metrics there:

```
python3 src/download.py streams --metrics https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/Bitstreams/Scenario-1-FHD/VTM/streams.csv /data/Origin/Scenario-1-FHD/VTM
```

#### metrics verification

to generate a verification `verify-metrics` command of the `metrics.py` script.

the following options are needed:
- `-o` : the path to the directory containing the original data to which localy computed results will be compared
- `-r` : the resulting report path is specific with

the metrics key on which the verification report should run are specified as the last arguments. eg `psnr y_psnr`
the following metrics keys are implemented: *psnr, y_psnr, u_psnr, v_psnr, ms_ssim, vmaf, bitrate, bitrate_log, wpsnr, y_wpsnr, u_wpsnr, v_wpsnr, psnr100, de100*

```
python3 ./src/metrics.py -c S1-VTM-01 verify-metrics
    -o /data/origin
    -r /data/origin/s1-vtm-01-verification-report.csv
    --doc - --email foo@corp.com --company corp 
    y_psnr psnr
```


_____

## docker usage

All the dependencies in a single docker image.

### building the docker images

on linux: `./build.sh` (should work on mac as well)
on windows: `.\build.ps1`

the build script builds several docker images:
- `vcc:base` used as provides common build environment, and metrics dependencies :  HDRTools, vmaf.
- `vcc:jm`, `vcc:hm`, `vcc:etm`, `vcc:ctm`, `vcc:aom` builds the respective reference encoders, decoders, and sei removal executable.
- `vcc:worker` copies pre-built binaries from the above images, supplies the appropriate environment variables, adds the scripts & config folders.

build a specific image : `docker image ls`
list docker images on your machine: `docker image ls`
remove a specific docker image: `docker image rm vcc:worker`
remove all docker image: `docker image rm vcc`


### using the worker image

The worker image is configured with `/data` as the root directory for all the `ReferenceSequences` and `Bitstreams` directories. 
When running the worker image; bitstreams and reference sequences must be mounted to the container /data directroy, eg. using docker's `--mount` option.

```
docker run -it --mount type=bind,source=/path/to/local/vcc_working_dir,target=/data" vcc:workers vcc.py -s S1-A01-264 decode
```

local `Bitstreams` and `ReferenceSequences` data may be organized arbitrarily and mounted to the appropriate directory:

```
docker run -it \
    --mount type=bind,source=/path/to/local/bitstreams_dir,target=/data/Bitstreams" \
    --mount type=bind,source=/path/to/local/sequences_dir,target=/data/ReferenceSequences" \
    vcc:workers vcc.py -s S1-A01-264 decode
```


### using docker-compose 

A sample docker-compose is provided. It starts a background worker and a redis queue. 
The `--queue` option of the `vcc.py` script can be used to add jobs to the queue.
The initial concurrency is set to 2 simultaneous jobs. Too high value for concurrency may not be suitable for all tasks (eg. 4K encodes). 

to use it, first configure local directories to mount using environment variables:
linux
``` 
export VCC_BITSTREAMS_DIR=/path/to/local/bitstreams_dir
export VCC_REFERENCES_DIR=/path/to/local/sequences_dir
```

windows
```
$env:VCC_BITSTREAMS_DIR="c:\path\to\local\bitstreams_dir"
$env:VCC_REFERENCES_DIR="c:\path\to\local\sequences_dir"
```

build workers:
```
docker-compose build workers
```

then start the stack:
```
docker-compose up -d
```

verify that all services are running:
```
docker-compose ps
```
the queue can be monitored with a web browser at [http://localhost:8888/](http://localhost:8888/)


queue some tasks, eg. decode all anchors encoded with s1-jm-01.cfg:
```
docker-compose exec worker python3 vcc.py --queue -c S1-JM-01 decode
```
