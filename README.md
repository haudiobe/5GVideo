
# Usage

```
./cmd.py ./anchor.json [encode] [decode] [metrics]
```

> `./cmd.py ./anchor.json` 

performs encoding, reconstruction and runs metrics computation in one step.

The tool assumes you have reference encoders compiled, with environment variables pointing to the executables.


# Supported encoders 

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
    "test_sequence": "path/to/references/sample.yuv",
    "encoder": "HM",
    "encoder_cfg": "path/to/anchor/encoder_cfg.cfg",
    "variants": {
        "variant_id0": {
            "OptionKey": "OptionValue",
            "OptionKey2": "{WORKING_DIR}/sub.cfg"
        },
        "variant_id1": {
            "OptionKey": "OptionValue"
        },
        [...]
    }
}
```

this configuration generates the following :
```
# coded bitstream
path/to/anchor/encoder_cfg.variant_id0.bin
path/to/anchor/encoder_cfg.variant_id0.enc.log

# reconstructed
path/to/anchor/encoder_cfg.variant_id0.yuv

# variant metrics
path/to/anchor/encoder_cfg.variant_id0.csv

[...]

path/to/anchor/encoder_cfg.variant_id1.bin
path/to/anchor/encoder_cfg.variant_id1.rec
path/to/anchor/encoder_cfg.variant_id1.enc.log
path/to/anchor/encoder_cfg.variant_id1.csv

[...]

# anchor metrics
path/to/anchor/encoder_cfg.csv
```

notes: 
- if **test_sequence** or **encoder_cfg** specify a relative path, it is relative to where the command is running
- **{WORKING_DIR}** resolves to the encoder config directory, eg. `path/to/anchor/sub.cfg`


# Raw video sequence descritpion

YUV sequences are currently described through a sidecar file.
eg. for the above `path/to/sample.yuv`, add the following `path/to/sample.json`

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

- metrics computation assumes reference sequence and reconstructed sequences share the same chroma format and bitdepth
- only planar YUV is supported at the moment
- color space conversions are left up to the encoder configuration


