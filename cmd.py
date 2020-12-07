#!/usr/bin/python3
import sys
import os
import csv

from anchor import AnchorCfg, VariantCfg, md5_checksum
from encoders import get_encoder
from metrics import VariantData, hdrtools_metrics, bd_q

from typing import List, Iterable

def to_csv(fout:str, fieldnames:List[str], values:List[dict]):
    with open(fout, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in values:
            writer.writerow(row)

def compute_variant_metrics(variant:VariantCfg) -> VariantData:
    duration = variant.anchor.frame_count * variant.anchor.reference.duration / variant.anchor.reference.framecount
    avg_bitrate = int(os.path.getsize(variant.bitstream) * 8 / duration)
    data = hdrtools_metrics(variant)
    to_dict = lambda keys, data : { k:v for k,v in zip(keys, data) }
    return VariantData(variant.basename, avg_bitrate, **to_dict(data['metrics'], data['avg']))

def compute_anchor_metrics(anchor:AnchorCfg):
    data = [compute_variant_metrics(v) for v in anchor.variants]
    if len(data) == 0:
        return
    keys = data[0].data.keys()
    to_csv(f'{anchor.working_dir / anchor.basename}.csv', keys, [m.data for m in data])
    return data

def encode_anchor(anchor:AnchorCfg, recon=True):
    enc = get_encoder(anchor.encoder_id)
    if enc == None:
        raise Exception(f'unknown encoder: {anchor.encoder_id}')
    for var in anchor.variants:
        enc.encode_variant(var, recon=recon)

def decode_anchor(anchor:AnchorCfg):
    enc = get_encoder(anchor.encoder_id)
    if enc == None:
        raise Exception(f'unknown encoder: {anchor.encoder_id}')
    for var in anchor.variants:
        enc.decode_variant(var)

def md5_reconstucted(anchor:AnchorCfg):
    for var in anchor.variants:
        h = md5_checksum(var.reconstructed)
        p = var.reconstructed.parent / f'{var.reconstructed.stem}.yuv.md5'
        with p.open('w') as f:
            f.write(h)

def md5_bitstream(anchor:AnchorCfg):
    for var in anchor.variants:
        h = md5_checksum(var.bitstream)
        p = var.bitstream.parent / f'{var.bitstream.stem}.md5'
        with p.open('w') as f:
            f.write(h)


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
        return None, False, False, False

    if not os.path.exists(sys.argv[1]):
        print(f'config file not found {sys.argv[1]}')
        return None, False, False, False
    
    cfg = sys.argv[1]

    if len(sys.argv) == 2:
        return cfg, True, True, True

    encode = "encode" in sys.argv
    decode = "decode" in sys.argv
    metrics = "metrics" in sys.argv

    if metrics:
        assert os.getenv('HDRMETRICS_TOOL') != None, 'HDRMETRICS_TOOL environment variable not set'
    print(metrics, os.getenv('HDRMETRICS_TOOL'))
    
    return cfg, encode, decode, metrics


def main():

    cfg, encode, decode, metrics = parse_args()

    if (cfg is None) or not (encode or decode or metrics):
        man()
        return
    
    anchor = AnchorCfg.load(cfg)

    if encode:
        encode_anchor(anchor, recon=decode)
        md5_bitstream(anchor)

    if decode and not encode:
        decode_anchor(anchor)

    if decode:
        md5_reconstucted(anchor)
    
    data = None

    if metrics:
        data = compute_anchor_metrics(anchor)
        for var in data:
            print(var.to_string())

if __name__ == "__main__":
    main()
