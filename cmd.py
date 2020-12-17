#!/usr/bin/python3
import sys
import os
import csv
import json

from anchor import AnchorTuple, VariantCfg, md5_checksum
from encoders import get_encoder
from metrics import VariantData, hdrtools_metrics, bd_q

from typing import List, Iterable

def save_variant_result(v:VariantCfg, m:VariantData):
        
        assert v.bitstream.exists(), "bitstream file not found"
        assert v.reconstructed.exists(), "reconsturcted file not found"
        bitstream_md5 = md5_checksum(v.bitstream)
        reconstruction_md5 = md5_checksum(v.reconstructed)
        metrics = m.to_dict()
        metrics.pop("Key", None)
        meta = {
            "Bitstream": {
                "URI": str(v.bitstream),
                "md5": bitstream_md5,
                "key": v.variant_id,
                "codecs": v.anchor.codecs
            },
            "Generation": {
                "sequence": str(v.anchor.reference.path),
                "encoder": v.anchor.encoder_id,
                "config-file": str(v.anchor.encoder_cfg),
                "config-variant": v.options,
                "log-file": str(v.encoder_log)
            },
            "Reconstruction": {
                "decoder": v.anchor.encoder_id,
                "log-file": str(v.decoder_log),
                "md5": reconstruction_md5
            },
            "Metrics": metrics
        }

        fp = v.anchor.working_dir / f'{v.basename}.json'
        with open(fp, 'w') as fo:
            json.dump(meta, fo, sort_keys=True, indent=4)


def to_csv(fout:str, fieldnames:List[str], values:List[dict]):
    with open(fout, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in values:
            writer.writerow(row)

def compute_variant_metrics(variant:VariantCfg) -> VariantData:
    metrics = hdrtools_metrics(variant)
    if variant.anchor.dry_run:
        return
    metrics["VMAF"] = None
    # Note: SEI data may be present/missing in the bitstream depending on the encoder configuration
    avg_bitrate = int(os.path.getsize(variant.bitstream) * 8 / variant.anchor.duration)
    data = VariantData(variant.variant_id, avg_bitrate, metrics)
    save_variant_result(variant, data)
    return data

def compute_anchor_metrics(anchor:AnchorTuple):
    data = [compute_variant_metrics(v) for v in anchor.variants]
    if len(data) == 0 or anchor.dry_run:
        return
    keys = data[0].get_keys()
    anchor_tupple = f'{anchor.working_dir / anchor.basename}.csv'
    to_csv(anchor_tupple, keys, [m.to_dict() for m in data])
    return anchor_tupple

def encode_anchor(anchor:AnchorTuple, recon=True):
    enc = get_encoder(anchor.encoder_id)
    if enc == None:
        raise Exception(f'unknown encoder: {anchor.encoder_id}')
    for var in anchor.variants:
        enc.encode_variant(var, recon=recon)

def decode_anchor(anchor:AnchorTuple):
    enc = get_encoder(anchor.encoder_id)
    if enc == None:
        raise Exception(f'unknown encoder: {anchor.encoder_id}')
    for var in anchor.variants:
        enc.decode_variant(var)

def man():
    h = """
    usage:
        cmd.py cfg.json [encode] [decode] [metrics]
    
    cfg.json
        a valid anchor configuration
    """
    print(h)


def parse_args():
    if len(sys.argv) <= 1:
        return None, False, False, False, False

    if not os.path.exists(sys.argv[1]):
        print(f'config file not found {sys.argv[1]}')
        return None, False, False, False, False
    
    cfg = sys.argv[1]

    if len(sys.argv) == 2:
        return cfg, True, True, True, False

    encode = "encode" in sys.argv
    decode = "decode" in sys.argv
    metrics = "metrics" in sys.argv
    dry_run = "-d" in sys.argv

    if metrics:
        assert os.getenv('HDRMETRICS_TOOL') != None, 'HDRMETRICS_TOOL environment variable not set'
    
    return cfg, encode, decode, metrics, dry_run


def main():

    cfg, encode, decode, metrics, dry_run = parse_args()

    if (cfg is None) or not (encode or decode or metrics):
        man()
        return
    
    anchor = AnchorTuple.load(cfg)
    anchor.dry_run = dry_run

    if encode:
        encode_anchor(anchor, recon=decode)

    if decode and not encode:
        decode_anchor(anchor)
    
    data = None

    if metrics:
        csv_data = compute_anchor_metrics(anchor)
        if anchor.dry_run:
            return
        print(f'\n\nanchor tupple metrics saved to: {csv_data}')

if __name__ == "__main__":
    main()
