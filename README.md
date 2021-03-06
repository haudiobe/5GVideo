
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

docker run --mount type=bind,source=/path/to/samples,target=/samples -it anchortools ./cmd.py /samples/anchors/sample_hm.json
```

> the --mount option mounts the directory *source* path, and makes it available as *target* path in the running container.

# Usage

```
./cmd.py ./anchor.json [encode] [decode] [metrics] [-d]
```

- the tool assumes you have reference encoders compiled, with environment variables pointing to the executables, see below.
- `-d` runs in dry mode, output the sequence of commands to stdout

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

metrics computation uses:
https://gitlab.com/standards/HDRTools


the following environment variables is needed:
```
HDRMETRICS_TOOL=/path/to/HDRTools/build/bin/HDRMetrics
```


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
    "reference": "path/to/reference/sample.json",
    "reference_segment": {
        "start_frame": 10,
        "frame_count": 30
    }
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


# Raw video sequence description

YUV sequences are currently described through a sidecar file.

the sidecar file format follows the json schema specified at : https://github.com/haudiobe/5G-Video-Content/blob/main/3gpp-raw-schema.json


# Current limitations
- framerate is converted to integer (HM only supports integer values)
- RGB support is not complete
- metrics computation assumes reference sequence and reconstructed sequences share the 'Properties'


