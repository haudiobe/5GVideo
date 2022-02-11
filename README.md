# Overview

these script can be used to produce / verify specific encoders and scenarios according to TR26.955.


## installation

```
git clone https://github.com/haudiobe/5GVideo 
cd 5GVideo
```

Please use a [python virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) to install dependencies and run the scripts. 

install dependencies:
```
pip3 install -r src/requirements.txt
```


### working directory

the scripts are meant to work on local directory containing data organized as in: https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/

the `VCC_WORKING_DIR` environment variable is used to configure the location of that directory.


### download content from server

downloads anchor bitstreams, metrics, encoder config, and reference sequence list for a given scenario:
```
python3 download.py streams https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/Bitstreams/Scenario-1-FHD/264/streams.csv $VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/264
```

downloads anchor reference sequences for a csv list:
```
python3 src/download.py sequences https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/Bitstreams/Scenario-1-FHD/264/streams.csv $VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/reference-sequences.csv
```


### characterization

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


### encode, decode, convert, metrics commands

`src/vcc.py` provides the commands to encode, decode, and compute metrics.

All the commands described below support processing a single anchor ...
```
python3 src/metrics.py -s S1-A11-265 csv-metrics
```

a single sequence (QP) ...
```
python3 src/metrics.py -s S1-A11-265-22 csv-metrics
```

or batch all anchors oif a given encoder config.
```
python3 src/metrics.py -c S1-HM-02 csv-metrics
```


#### encode

```
JM19_0_ENCODER=~/deps/JM/bin/lencod_static

python3 src/vcc.py -s S1-A01-264 encode [--dry-run]
```

*dry-run* mode prints the command used to encode to stdout but doesn't run it, but it does compute existing bitstream md5.


#### decode

```
JM19_0_DECODER=~/deps/JM/bin/ldecod_static

python3 src/vcc.py -s S1-A01-264 decode [--dry-run]
```

*dry-run* mode prints the command used to encode to stdout but doesn't run it, but it does compute existing reconstruction md5.


#### convert

```
HDRCONVERT_TOOL=~/deps/HDRTools/build/bin/HDRConvert

python3 src/vcc.py -s S1-A11-265 decode --reference [--dry-run]
python3 src/vcc.py -s S1-A11-265 decode --reconstructions [--dry-run]
```

Performs conversion needed before metrics computation using HDRTools' HDRConvert.

when --reconstructions is specfied, the reconstructions for all QPs are processed. 
when --reference is specified, the reference sequence is processed instead.

Conversion is either be 8 to 10 bit YUV, or YUV to EXR. When conversion is not needed for a given sequence, the script does nothing.

The converted sequences are stored in a tmp subfolder for each anchor.

*dry-run* mode prints the command used to convert to stdout but doesn't run it.


#### metrics

```
HDRMETRICS_TOOL=~/deps/HDRTools/build/bin/HDRMetrics
VMAF_EXE=~/deps/vmaf/libvmaf/build/tools/vmafossexec
VMAF_MODEL=/home/deps/vmaf/model/vmaf_v0.6.1.json

python3 src/vcc.py -s S1-A11-265 metrics [--dry-run]
```

Computes metrics for the given sequence. This step also computes bitrate after removing SEI.

HDRTOOLS or VMAF can be enabled / disabled through environment variables.
```
DISABLE_HDRMETRICS=1
DISABLE_VMAF=1
```

This step is meant to run after successfull `conversion`. It uses the converted sequence where appropriate, and raise an error if it doesn't exist.

*dry-run* mode doesn't run HDRTools or vmaf, but it does parse existing logs.


##### metrics csv export

Metrics are saved on the anchors JSON file. A script can be used to export them to .csv:

```
python3 src/metrics.py -s S1-A11-265 csv-metrics
python3 src/metrics.py -c S1-HM-02 csv-metrics
```

For the above commands, the metrics `.csv` files can then be found in: `$VCC_WORKING_DIR/Bitstreams/Scenario-1-FHD/265/Metrics`
