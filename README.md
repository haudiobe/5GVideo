
# Quickstart

Given an anchor definition, the tool performs encoding, reconstruction and runs additional metrics computation for all variants.

1. add the `samples/references/yuv420_1280x720_8bit_rec709.yuv` sequence

2. set the environment variable: `HM_ENCODER=/path/to/HM/bin/TAppEncoderStatic`

3. generate the sample HM anchor 
```
./cmd.py ./samples/anchors/sample_hm.json encode decode`
```

## Using docker 

a sample [Dockerfile](https://docs.docker.com/get-docker/) is provided to build an image containing all the dependencies : HM, JM, and VTM

```
git clone https://github.com/haudiobe/5GVideo.git

cd 5GVideo

docker build -t anchortools -f ./docker/Dockerfile .

# add the missing `./samples/references/yuv420_1280x720_8bit_rec709.yuv` sequence

docker run --mount type=bind,source=./samples,target=/samples -it anchortools cmd.py /samples/anchors/sample_hm.json
```

> the --mount option mounts the directory *source* path, and makes it available as *target* path in the running container.

# Usage

```
./cmd.py ./anchor.json [encode] [decode] [metrics]
```

- the tool assumes you have reference encoders compiled, with environment variables pointing to the executables, see below.


## encode

```
./cmd.py ./anchor.json encode decode
```

runs the **reference encoder** to generate both bitstream, and reconstructed sequence for all the anchors, with the reconstructed chroma format as specified in the encoder config.

```
./cmd.py ./anchor.json encode
```

runs the reference encoder to generate bitstream only.


## decode

```
./cmd.py ./anchor.json decode
```

runs the **reference decoder** to reconstruct the bitstream, with the output chroma format matching the bitstream.


## metrics 

```
./cmd.py ./anchor.json metrics
```

generates some metrics for each variant defined in the anchor

> the `metrics` options uses third party tools, see below.



# Reference encoders 

the following environment variables are needed depending on the encoder/decoder you want to use:

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

[VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)
```
VTM_ENCODER=/path/to/bin/EncoderAppStatic
VTM_DECODER=/path/to/bin/DecoderAppStatic
```

### Adding a custom encoder

implement the ReferenceEncoder interface (encoder_id, encode_variant, decode_variant) and decorate your class (@register_encoder)




# Anchor definition

example ***anchor.json*** :
```
{
    "description": "human readable description, use case, settings overview ...",
    "reference": "path/to/reference/sample.yuv",
    "encoder": "HM",
    "encoder_cfg": "path/to/anchor/encoder_cfg.cfg",
    "variants": {
        "variant_id0": {
            "OptionKey": "OptionValue0",
            "OptionKey2": "{ANCHOR_DIR}/sub0.cfg",
            [...]
        },
        "variant_id1": {
            "OptionKey": "OptionValue1",
            "OptionKey2": "{ANCHOR_DIR}/sub1.cfg",
            [...]
        },
        [...]
    }
}
```

### notes :
- if `test_sequence` or `encoder_cfg` is a relative path, it is interpreted as **relative to *anchor.json***.
- when used in the variant options `{ANCHOR_DIR}` is expanded to the anchor's encoder config directory: **path/to/anchor**.

## Variant options to encoder CLI args mapping

### **HM**, **VTM**
```
{
    "encoder_cfg": "/encoder.cfg",
    "variants": {
        "variant_id0": {
            "-k": "v",
            "--Key": "value"
        }
    }
}
```
maps to `-c /encoder.cfg -k v --Key=value`

### **JM**
```
{
    "encoder_cfg": "/encoder.cfg",
    "variants": {
        "variant_id0": {
            "-f": "file.cfg",
            "Key": "value"
        }
    }
}
```
maps to `-d /encoder.cfg -f file.cfg -p Key=value`




## **Output**

the above configuration would generate the following :
```
# variant bitstream
path/to/anchor/encoder_cfg.variant_id0.bit

# encoder log (VTM has additional .opl file)
path/to/anchor/encoder_cfg.variant_id0.enc.log

# reconstructed variant
path/to/anchor/encoder_cfg.variant_id0.yuv

# variant metrics
path/to/anchor/encoder_cfg.variant_id0.csv

[...]

path/to/anchor/encoder_cfg.variant_id1.bit
path/to/anchor/encoder_cfg.variant_id1.yuv
path/to/anchor/encoder_cfg.variant_id1.enc.log
path/to/anchor/encoder_cfg.variant_id1.csv

[...]

# averaged metrics, one variant per row
path/to/anchor/encoder_cfg.csv
```


# Raw video sequence descritpion

YUV sequences are currently described through a sidecar file.

eg. for the above `path/to/reference/sample.yuv`, add the following `path/to/reference/sample.json`

```
{
    "width": 1280,
    "height": 720,
    "chroma_format": "yuv",
    "chroma_subsampling": "420",
    "bitdepth": 8,
    "fps": 30,
    "color_space": "rec709",
    "transfer": null,
    "framecount": 30
}
```

# Current limitations
- fps is converted to integer
- only planar YUV reference sequences are supported
- the **transfer** and **color_space** properties are currently ignored, however color space conversions can be configured through the encoder.cfg file and through variant options for each encoder
- metrics computation assumes reference sequence and reconstructed sequences share the same chroma format, frame packing and bitdepth


# Dependencies [metrics]

The open-source gpac application is needed for metrics computation.

Detailed build instruction please refer to : https://github.com/gpac/gpac/wiki/Build-Introduction

the path to the gpac executable can be configured through environment variable, eg. :
```
GPAC_APP=/path/to/bin/gpac
```


