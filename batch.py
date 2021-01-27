#!/usr/bin/python3
from pathlib import Path
import json
import subprocess
import argparse

ENCODER_IDS = ('JM', 'ETM') # ('JM', 'HM', 'VTM', 'ETM')

QP = [ 12, 15, 18, 23, 24, 27, 30, 33, 36 ]

CFG_TEMPLATE = {
    "reference": "../references/yuv420_1280x720_8bit_rec709.json",
    "reference_segment": {
        "start_frame": 0,
        "frame_count": 1
    },
}

def gen_variants(encoder_id, QPs):
    variants = {}
    key = "q"
    if encoder_id == "JM":
        key = "QPISlice"
    elif encoder_id in ["HM","VTM"]:
        key = "--QP"
    elif encoder_id == "ETM":
        key = "-q"
    for qp in QPs:
        variants[f'qp{qp}'] = { key: qp }
    return variants

def iter_configs():
    for e in ENCODER_IDS:
        cfg = CFG_TEMPLATE.copy()
        cfg["encoder_id"] = e
        cfg["encoder_cfg"] = f'sample_{e.lower()}/sample.{e.lower()}.cfg'
        cfg['variants'] = gen_variants(e, QP)
        yield cfg, f'./samples/anchors/sample_{e.lower()}.json'

def encode_all(filenames, dry=False):
    for f in filenames:
        print('# encoding ', f)
        cmd = ['python3', './cmd.py', f, 'encode']
        if dry:
            cmd.append('-d')
        subprocess.run(cmd)

def decode_all(filenames, dry=False):
    for f in filenames:
        print('# decoding ', f)
        cmd = ['python3', './cmd.py', f, 'decode']
        if dry:
            cmd.append('-d')
        subprocess.run(cmd)

def metrics_all(filenames, dry=False):
    for f in filenames:
        print('# computing metrics ', f)
        cmd = ['python3', './cmd.py', f, 'metrics']
        if dry:
            cmd.append('-d')
        subprocess.run(cmd)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='batch')
    parser.add_argument('-c', '--cfg', action='store_true', required=False)
    parser.add_argument('-e', '--encode', action='store_true', required=False)
    parser.add_argument('-d', '--decode', action='store_true', required=False)
    parser.add_argument('-m', '--metrics', action='store_true', required=False)
    parser.add_argument('--dry', action='store_true', required=False)
    args = parser.parse_args()
    
    noop = True
    filenames = []

    for data, filename in iter_configs():
        if args.cfg:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                print('# generated ', filename)
                noop = False
            assert not noop, 'no config'
        filenames.append(filename)

    if args.encode:
        encode_all(filenames, args.dry)
        noop = False

    if args.decode:
        encode_all(filenames, args.dry)
        noop = False

    if args.metrics:
        metrics_all(filenames, args.dry)
        noop = False

    if noop:
        parser.print_help()

