
# Usage

```
./cmd.py ./anchor.json [encode] [decode] [metrics]
```

given an anchor definition, the tool performs encoding, reconstruction and runs additional metrics computation in one step for all variants.

> the tool assumes you have reference encoders compiled, with environment variables pointing to the executables, see below.


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

> the `metrics` options currently requires third party tools, see below.



# Reference encoders 

the following environment variables are needed depending on the encoder/decoder you want to use:
```
JM_ENCODER=/path/to/JM/bin/lencod_static
JM_DECODER=/path/to/JM/bin/ldecod_static

HM_ENCODER=/path/to/HM/bin/TAppEncoderStatic
HM_DECODER=/path/to/bin/TAppDecoderStatic

VTM_ENCODER=/path/to/bin/EncoderAppStatic
VTM_DECODER=/path/to/bin/DecoderAppStatic
```

### Adding a custom encoder

implement the ReferenceEncoder interface (encoder_id, encode_variant, decode_variant) and decorate your class (@register_encoder)




# Anchor definition

```
{
    "description": "human readable description, use case, settings overview ...",
    "test_sequence": "path/to/reference/sample.yuv",
    "encoder": "HM",
    "encoder_cfg": "path/to/anchor/encoder_cfg.cfg",
    "variants": {
        "variant_id0": {
            "OptionKey": "OptionValue",
            "OptionKey2": "{ANCHOR_DIR}/sub.cfg"
        },
        "variant_id1": {
            "OptionKey": "OptionValue"
            "OptionKey2": "{ANCHOR_DIR}/sub.cfg"
        },
        [...]
    }
}
```

notes: 
- if `test_sequence` or `encoder_cfg` is a relative path, it is interpreted as **relative the current shell working directory**
- when used, `{ANCHOR_DIR}` resolves to the anchor's encoder config directory, **path/to/anchor**

the above configuration would generate the following :
```
# variant bitstream
path/to/anchor/encoder_cfg.variant_id0.bit
path/to/anchor/encoder_cfg.variant_id0.enc.log

# reconstructed variant
path/to/anchor/encoder_cfg.variant_id0.yuv

# variant metrics
path/to/anchor/encoder_cfg.variant_id0.csv

[...]

path/to/anchor/encoder_cfg.variant_id1.bit
path/to/anchor/encoder_cfg.variant_id1.rec
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
    "framecount": 90
}
```

# Current limitations
- transfer is not supported, color space conversions is left up to the encoder configuration
- only planar YUV is supported at the moment
- metrics computation assumes reference sequence and reconstructed sequences share the same chroma format and bitdepth



