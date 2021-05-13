#!/usr/bin/python3

import argparse
from pathlib import Path

from typing import Iterable

from encoders import get_encoder
from metrics import compute_metrics
from anchor import AnchorTuple, ReconstructionMeta, reference_sequences_dict, iter_anchors, iter_variants

def compute_anchor_metrics(*anchors:Iterable[AnchorTuple], decode=True, overwrite=False, dry_run=False):
    for a in anchors:
        a.dry_run = dry_run
        enc = get_encoder(a.encoder_id)
        assert enc != None, f'unknown encoder {a.encoder_id}'
        for vf, vd in iter_variants(a):
            assert vd != None, 'bitstream metadata not found'
            if (vd.metrics != None) and (not overwrite):
                print('# skipping', vf, ' - use -y to overwrite')
                continue
            if decode:
                recon = enc.decode_variant(a, vd.variant_id, vd.variant_cli)
                vd.reconstruction = recon.to_dict()
            else:
                recon = ReconstructionMeta(a.encoder_id, a.working_dir / f'{vd.variant_id}.yuv', Path(vd.reconstruction['log-file']), md5=False)
            m = compute_metrics(a, vd, recon)
            vd.metrics = m.to_dict()
            vd.save_as(vf)


def encode_anchor_bitstreams(*anchors:Iterable[AnchorTuple], decode=True, overwrite=False, dry_run=False):
    for a in anchors:
        a.dry_run = dry_run
        enc = get_encoder(a.encoder_id)
        assert enc != None, f'unknown encoder {a.encoder_id}'
        
        for vf, vd in iter_variants(a):
            if (vd != None) and (not overwrite):
                print('# skipping', vf, ' already exists. use -y to overwrite')
                continue
            vd = enc.encode_variant(a, vd.variant_id, vd.variant_cli)
            vd.save_as(vf)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', help='"encoder" or "decoder"')
    parser.add_argument('--scenario_dir', required=True, type=str, help='scenario directory')
    parser.add_argument('-k', '--key', required=False, type=str, default=None, help='an optional anchor key')
    parser.add_argument('-y', '--overwrite', required=False, action='store_true', default=False, help='overwrite if data already exists')
    parser.add_argument('--dry-run', required=False, action='store_true', default=False)
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)
    assert scenario_dir.is_dir(), f'invalid scenario directory {scenario_dir}'
    
    anchors_csv = scenario_dir / 'anchors.csv'
    assert anchors_csv.exists(), f'anchor list not found {anchors_csv.resolve()}'
    
    references_csv = scenario_dir / 'reference-sequence.csv'
    assert references_csv.exists(), f'reference list not found {references_csv.resolve()}'

    sequences_dir = scenario_dir / '../../ReferenceSequences'
    assert sequences_dir.is_dir(), f'sequence directory not found {sequences_dir.resolve()}'
    
    cfg_dir = scenario_dir / '../CFG'
    if args.cmd == "encoder":
        assert cfg_dir.exists(), f'config dir not found {cfg_dir.resolve()}'
    elif args.cmd not in ["decoder", "metrics"]:
        parser.error(f'invalid command "{args.cmd}" must be one of encoder, decoder')
    
    anchor_keys = None
    if args.key != None:
        anchor_keys = [args.key]

    return args.cmd, scenario_dir, anchors_csv, anchor_keys, references_csv, sequences_dir, cfg_dir, args.dry_run, args.overwrite


def main():

    cmd, scenario_dir, anchors_csv, anchor_keys, references_csv, sequences_dir, cfg_dir, dry_run, y = parse_args()
    refs = reference_sequences_dict(references_csv, sequences_dir)
    anchors = iter_anchors(anchors_csv, refs, scenario_dir, cfg_dir, keys=anchor_keys)

    if cmd == "decoder":
        compute_anchor_metrics(*anchors, overwrite=y, dry_run=dry_run)
    elif cmd == "metrics":
        compute_anchor_metrics(*anchors, decode=False, overwrite=y, dry_run=dry_run)
    elif cmd == "encoder":
        encode_anchor_bitstreams(*anchors, overwrite=y, dry_run=dry_run)


if __name__ == "__main__":
    main()