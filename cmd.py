#!/usr/bin/python
import sys
import os
import csv

from anchor import AnchorCfg, VariantCfg
from encoders import get_encoder
from metrics import VariantData, psnr_stats, ssim_stats, avg, bd_q

from typing import List, Iterable

def to_csv(fout:str, fieldnames:List[str], values:List[dict]):
    with open(fout, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in values:
            writer.writerow(row)

metrics_keys = [
    'psnr_y',
    'psnr_u',
    'psnr_v',
    'psnr_avg',
    'mse_y',
    'mse_u',
    'mse_v',
    'mse_avg',
    'ssim_y',
    'ssim_u',
    'ssim_v',
    'ssim_all'
]

def compute_variant_metrics(variant:VariantCfg) -> VariantData:
    avg_bitrate = os.path.getsize(variant.bitstream) * 8 / variant.anchor.reference.duration
    v = []
    for frame in zip(psnr_stats(variant), ssim_stats(variant)):
        f = {}
        for metric in frame:
            f.update(metric)
        v.append(f)
    to_csv(f'{variant.anchor.working_dir / variant.basename}.csv', metrics_keys, v)
    return VariantData(variant.basename, avg_bitrate, **avg(v, *metrics_keys))

def compute_anchor_metrics(anchor:AnchorCfg):
    data = [compute_variant_metrics(v) for v in anchor.variants]
    if len(data) == 0:
        return
    keys = data[0].data.keys()
    to_csv(f'{anchor.working_dir / anchor.basename}.csv', keys, [m.data for m in data])
    return data

def encode_anchor(anchor:AnchorCfg, baseline=True, save_reconstructed=True):
    enc = get_encoder(anchor.encoder_id)
    if enc == None:
        raise Exception(f'unknown encoder: {anchor.encoder_id}')
    if baseline == True:
        var = VariantCfg(anchor, 'default', {})
        enc.encode_variant(var, save_reconstructed=True)
        return
    for var in anchor.variants:
        enc.encode_variant(var, save_reconstructed=True)


def decode_anchor(anchor:AnchorCfg, baseline=True):
    enc = get_encoder(anchor.encoder_id)
    if enc == None:
        raise Exception(f'unknown encoder: {anchor.encoder_id}')
    if baseline == True:
        var = VariantCfg(anchor, 'default', {})
        enc.decode_variant(var)
        return
    for var in anchor.variants:
        enc.decode_variant(var)


def man():
    h = """
    usage:
        cmd.py cfg.json [encode|decode|metrics|compare]
    
    cfg.json
        a valid anchor configuration
    """
    print(h)


def parse_args():
    if len(sys.argv) <= 1:
        return None, False, False, False, False, False

    if not os.path.exists(sys.argv[1]):
        print(f'config file not found {sys.argv[1]}')
        return None, False, False, False, False, False
    
    cfg = sys.argv[1]
    
    if len(sys.argv) == 2:
        return cfg, False, True, True, True, True

    encode = "encode" in sys.argv
    decode = "decode" in sys.argv
    metrics = "metrics" in sys.argv
    compare = "compare" in sys.argv
    baseline = "baseline" in sys.argv
    
    return cfg, baseline, encode, decode, metrics, compare


def main():

    cfg, baseline, encode, decode, metrics, compare = parse_args()

    if (cfg is None) or not (encode or decode or metrics or compare):
        man()
        return
    
    anchor = AnchorCfg.load(cfg)
    # @TODO: anchor.working_dir = "/path/to/custom/dir"

    if encode:
        encode_anchor(anchor, baseline, save_reconstructed=decode)

    if decode and not encode:
        decode_anchor(anchor, baseline)

    data = None

    if metrics:
        data = compute_anchor_metrics(anchor)
        for var in data:
            print(var.to_string())

    if compare:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
