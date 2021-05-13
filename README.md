
# Quickstart

* the script are designed to work with data downloaded from: `https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo`


## Using docker

a sample [Dockerfile](https://docs.docker.com/get-docker/) is provided to build an image with the scripts dependencies.


to build the image:
```
git clone https://github.com/haudiobe/5GVideo.git
cd 5GVideo
docker build -t anchortools -f ./docker/Dockerfile .
```

to use the image:
```
root_dir=/path/to/host/data
scenario=Scenario-3
anchor_key=S3-A36-265

docker run -it \
    --mount type=bind,source=$root_dir/Anchors,target=/data/Anchors \
    --mount type=bind,source=$root_dir/ReferenceSequences,target=/data/ReferenceSequences \
    anchortools:latest ./verify.py decoder --scenario_dir /data/Anchors/$scenario -k $anchor_key
```


## anchor verification 

> see the `samples` directory for sample verification logs and reports. 

`verify.py` script runs verification for bitstream or metrics, updates the anchor's metrics if verification is successfull.

verify a specific anchor's bitstreams:
```
verify.py --scenario_dir /data/Anchors/Scenario-3 -k S3-A36-265 bitstream
```

verify all bitstreams in a scenarios:
```
verify.py --scenario_dir /data/Anchors/Scenario-3 -k S3-A36-265 bitstream
```

verify a specific anchor's metrics:
```
verify.py --scenario_dir /data/Anchors/Scenario-3 -k S3-A36-265 bitstream
```

verify all metrics in a scenarios:
```
verify.py --scenario_dir /data/Anchors/Scenario-3 -k S3-A36-265 bitstream
```


**Report template**

the `--template` argument can be used to provide the path to a json template for the report (contact info, etc ...), eg.:
```
verify.py --scenario_dir /data/Anchors/Scenario-3 -k S3-A36-265 bitstream --template ./report-template.json
```

**Custom directory layout**

the verification script has additional command line arguments to customize directory layout, see `verify.py -h`.



## metrics generation

decode and compute metrics for a specific anchor, eg. S3-A36-265 in Scenario-3: 

`cmd.py --scenario_dir /data/Anchors/Scenario-3 -k S3-A36-265 decoder`

decode and compute metrics for all anchors in Scenario-3: 

`cmd.py --scenario_dir /data/Anchors/Scenario-3 decoder`

**Notes**

* reference sequences and encoder configuration should be accessible from the same relative path, relative to the target scenario. 
* the scenario directory must contain `anchors.csv`, `reference-sequence.csv`

## anchor generation

encode a specific anchor, eg. S3-A36-265 in Scenario-3: 
```
create.py --scenario_dir /data/Anchors/Scenario-3 -k S3-A36-265 encoder
```

encode all anchors in Scenario-3: 
```
create.py --scenario_dir /data/Anchors/Scenario-3 encoder
```

**Notes**

* reference sequences and encoder configuration should be accessible from the same relative path, relative to the target scenario. 
* the scenario directory must contain `anchors.csv`, `reference-sequence.csv`


# environment variables 

Some environment variables are configured in the docker file.
In order to run the scirpts on a custom environment, the following environment variables are used to locate the executables.

## reference encoder/decoders:

[JM](https://vcgit.hhi.fraunhofer.de/jct-vc/JM)
```
JM_ENCODER=/path/to/JM/bin/lencod_static
JM_DECODER=/path/to/JM/bin/ldecod_static
```

[HM](https://vcgit.hhi.fraunhofer.de/jct-vc/HM)
```
HM_ENCODER=/path/to/HM/bin/TAppEncoderStatic
HM_DECODER=/path/to/bin/TAppDecoderStatic
```

[SCM](https://vcgit.hhi.fraunhofer.de/jvet/HM/-/tree/HM-SCC-extensions)
ENV SCM_ENCODER=/path/to/SCM/bin/TAppEncoderStatic
ENV SCM_DECODER=/path/to/SCM/bin/TAppDecoderStatic

[VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
```
VTM_ENCODER=/path/to/bin/EncoderAppStatic
VTM_DECODER=/path/to/bin/DecoderAppStatic
```

### Adding a custom encoder
implement the ReferenceEncoder interface (encoder_id, encode_variant, decode_variant) and decorate your class (@register_encoder)


### metrics computation:

[HDRTools](https://gitlab.com/standards/HDRTools)
```
HDRMETRICS_TOOL=/path/to/HDRTools/build/bin/HDRMetrics
```

[VMAF](https://github.com/Netflix/vmaf)
```
VMAF_EXE=/path/to/vmaf/libvmaf/build/tools/vmaf
VMAF_MODEL=path=/path/to/vmaf/model/vmaf_v0.6.1.json:enable_transform
```

# Raw video sequence description
YUV sequences are currently described through a sidecar file.
the sidecar file format follows the json schema specified at : https://github.com/haudiobe/5G-Video-Content/blob/main/3gpp-raw-schema.json


# Limitations
- framerate is converted to integer (HM only supports integer values)
- RGB support is not complete
- HDR PQ metrics are not complete 

